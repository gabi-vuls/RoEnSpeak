/* roenspeak_whisper.c - ctypes-friendly C ABI over whisper.cpp
 *
 * Compiles BOTH as C11 and as C++ (clang++ -x c++). Build it as C++ to get the
 * try/catch guard around whisper.cpp calls (whisper_full is NOT exception-safe
 * upstream: only model load is guarded, so std::bad_alloc from whisper_full /
 * whisper_full's internal vectors can escape into a C caller and abort the
 * process). The C build is provided for reference / minimal-runtime builds.
 */

#include "roenspeak_whisper.h"

#include "whisper.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <signal.h>

#ifndef RS_WHISPER_BUILD_COMMIT
#  define RS_WHISPER_BUILD_COMMIT ""
#endif

#if defined(__cplusplus)
#  define RS_TLS thread_local
#  include <new>
#  include <exception>
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#  define RS_TLS _Thread_local
#else
#  define RS_TLS __thread
#endif

/* ------------------------------------------------------------------ errors */

#define RS_ERRLEN 1024
static RS_TLS char g_err[RS_ERRLEN] = {0};
/* 0 silent, 1 errors, 2 all. Aligned word, written rarely (once at startup),
 * read from whatever thread ggml logs on: a plain load/store is fine. */
static int               g_log_level = 1;
static rs_whisper_log_fn g_log_sink  = NULL;  /* NULL => stderr */
static void *            g_log_user  = NULL;

/* Emit one already-formatted fragment to wherever logging currently goes. */
static void rs_emit(int level, const char * text) {
    rs_whisper_log_fn sink = g_log_sink;
    if (sink) {
        sink(level, text, g_log_user);
    } else {
        fputs(text, stderr);
    }
}

static void rs_set_err(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_err, RS_ERRLEN, fmt, ap);
    va_end(ap);
    if (g_log_level >= 1) {
        char line[RS_ERRLEN + 32];
        snprintf(line, sizeof(line), "roenspeak_whisper: %s\n", g_err);
        rs_emit((int) GGML_LOG_LEVEL_ERROR, line);
    }
}

const char * rs_whisper_last_error(void) { return g_err; }

static void rs_log_cb(enum ggml_log_level level, const char * text, void * user) {
    (void) user;
    if (!text) return;
    if (level == GGML_LOG_LEVEL_ERROR) {
        /* keep the newest whisper/ggml error line reachable from Python */
        size_t n = strlen(text);
        while (n > 0 && (text[n-1] == '\n' || text[n-1] == '\r')) n--;
        if (n >= RS_ERRLEN) n = RS_ERRLEN - 1;
        memcpy(g_err, text, n);
        g_err[n] = '\0';
    }
    if (g_log_level >= 2 || (g_log_level >= 1 && level == GGML_LOG_LEVEL_ERROR)) {
        rs_emit((int) level, text);
    }
}

/* Installing the callbacks is idempotent; both setters are global by design. */
static void rs_install_log_cb(void) {
    whisper_log_set(rs_log_cb, NULL);
    ggml_log_set(rs_log_cb, NULL);
}

void rs_whisper_set_log_level(int level) {
    g_log_level = level;
    rs_install_log_cb();
}

void rs_whisper_set_log_sink(rs_whisper_log_fn fn, void * user) {
    g_log_user = user;
    g_log_sink = fn;   /* publish the pointer last */
    rs_install_log_cb();
}

/* Capture whisper/ggml logging from the moment the dylib is loaded, so that
 * nothing (not even a backend-registration warning emitted by a static
 * initializer in a sibling dylib, or rs_whisper_devices() before any model is
 * loaded) can reach stdout/stderr unfiltered in a bundled .app. */
__attribute__((constructor)) static void rs_whisper_ctor(void) { rs_install_log_cb(); }

/* ------------------------------------------------------------- dyn strings */

typedef struct { char * p; size_t len, cap; } rs_str;

static void rs_str_free(rs_str * s) { free(s->p); s->p = NULL; s->len = s->cap = 0; }

static int rs_str_append(rs_str * s, const char * add) {
    size_t n = strlen(add);
    if (s->len + n + 1 > s->cap) {
        size_t cap = s->cap ? s->cap : 256;
        while (cap < s->len + n + 1) cap *= 2;
        char * q = (char *) realloc(s->p, cap);
        if (!q) return -1;
        s->p = q; s->cap = cap;
        if (s->len == 0) s->p[0] = '\0';
    }
    memcpy(s->p + s->len, add, n + 1);
    s->len += n;
    return 0;
}

static void rs_str_clear(rs_str * s) { s->len = 0; if (s->p) s->p[0] = '\0'; }

/* ----------------------------------------------------------------- handle */

typedef struct rs_ctx {
    struct whisper_context * ctx;
    pthread_mutex_t          mu;

    int   req_gpu;
    int   has_gpu;
    char  backend[16];

    /* sticky knobs, merged into whisper_full_params on every run */
    struct whisper_full_params opt;  /* template; strategy/lang/etc overridden per run */
    char * initial_prompt;
    char * suppress_regex;

    /* results of the last run */
    int    n_segments;
    float  avg_logprob;
    float  max_nsp;
    char   lang[16];
    double run_ms, audio_ms, prep_ms;
    double t_run_start;
    rs_str full_text;
    float * seg_logprob;      /* n_segments floats */
    int     seg_logprob_cap;

    volatile sig_atomic_t abort_flag;
} rs_ctx;

static bool rs_encoder_begin_cb(struct whisper_context * ctx, struct whisper_state * st, void * user);

static bool rs_abort_cb(void * user) {
    rs_ctx * h = (rs_ctx *) user;
    return h->abort_flag != 0;   /* true => abort graph computation */
}

/* ----------------------------------------------------------- library info */

int          rs_whisper_abi_version(void)      { return RS_WHISPER_ABI_VERSION; }
const char * rs_whisper_whisper_version(void)  { return whisper_version(); }
const char * rs_whisper_build_commit(void)     { return RS_WHISPER_BUILD_COMMIT; }
const char * rs_whisper_system_info(void)      { return whisper_print_system_info(); }
int          rs_whisper_lang_id(const char * l){ return (l && *l) ? whisper_lang_id(l) : -1; }

static int rs_gpu_dev_present(void) {
    size_t n = ggml_backend_dev_count();
    for (size_t i = 0; i < n; ++i) {
        enum ggml_backend_dev_type t = ggml_backend_dev_type(ggml_backend_dev_get(i));
        if (t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU) return 1;
    }
    return 0;
}

int rs_whisper_metal_available(void) { return rs_gpu_dev_present(); }

const char * rs_whisper_devices(void) {
    static char buf[1024];
    static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_lock(&mu);
    buf[0] = '\0';
    size_t n = ggml_backend_dev_count();
    for (size_t i = 0; i < n; ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        size_t fr = 0, tot = 0;
        ggml_backend_dev_memory(d, &fr, &tot);
        const char * tn = "other";
        switch (ggml_backend_dev_type(d)) {
            case GGML_BACKEND_DEVICE_TYPE_CPU:   tn = "cpu";   break;
            case GGML_BACKEND_DEVICE_TYPE_GPU:   tn = "gpu";   break;
            case GGML_BACKEND_DEVICE_TYPE_IGPU:  tn = "igpu";  break;
            case GGML_BACKEND_DEVICE_TYPE_ACCEL: tn = "accel"; break;
            default: break;
        }
        char one[256];
        snprintf(one, sizeof(one), "%s%s:%s(type=%s,free=%zuMiB,total=%zuMiB)",
                 i ? ", " : "", ggml_backend_dev_name(d),
                 ggml_backend_dev_description(d), tn, fr >> 20, tot >> 20);
        strncat(buf, one, sizeof(buf) - strlen(buf) - 1);
    }
    pthread_mutex_unlock(&mu);
    return buf;
}

/* ------------------------------------------------------------- init/free */

static void rs_init_opts(rs_ctx * h) {
    h->opt = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    /* quiet, deterministic, dictation-friendly defaults */
    h->opt.print_progress   = false;
    h->opt.print_realtime   = false;
    h->opt.print_timestamps = false;
    h->opt.print_special    = false;
    h->opt.no_context       = true;   /* no cross-utterance prompt carry-over */
    h->opt.temperature      = 0.0f;
    h->opt.temperature_inc  = 0.0f;   /* temperature fallback OFF */
    h->opt.suppress_blank   = true;
    h->opt.no_timestamps    = false;
    h->opt.single_segment   = false;
}

void * rs_whisper_init_ex(const char * model_path, int use_gpu, int flash_attn) {
    g_err[0] = '\0';
    if (!model_path || !*model_path) { rs_set_err("model_path is empty"); return NULL; }

    rs_ctx * h = (rs_ctx *) calloc(1, sizeof(rs_ctx));
    if (!h) { rs_set_err("out of memory"); return NULL; }
    if (pthread_mutex_init(&h->mu, NULL) != 0) { free(h); rs_set_err("mutex init failed"); return NULL; }

    struct whisper_context_params cp = whisper_context_default_params();
    cp.use_gpu    = use_gpu    ? true : false;
    cp.flash_attn = flash_attn ? true : false;
    cp.gpu_device = 0;

#ifdef __cplusplus
    try {
#endif
        h->ctx = whisper_init_from_file_with_params(model_path, cp);
#ifdef __cplusplus
    } catch (const std::exception & e) {
        rs_set_err("exception loading model: %s", e.what()); h->ctx = NULL;
    } catch (...) {
        rs_set_err("unknown exception loading model"); h->ctx = NULL;
    }
#endif
    if (!h->ctx) {
        if (!g_err[0]) rs_set_err("whisper_init_from_file_with_params failed for '%s'", model_path);
        pthread_mutex_destroy(&h->mu);
        free(h);
        return NULL;
    }

    h->req_gpu = use_gpu ? 1 : 0;
    h->has_gpu = (use_gpu && rs_gpu_dev_present()) ? 1 : 0;
    snprintf(h->backend, sizeof(h->backend), "%s", h->has_gpu ? "Metal" : "CPU");
    rs_init_opts(h);
    return (void *) h;
}

void * rs_whisper_init(const char * model_path, int use_gpu) {
    return rs_whisper_init_ex(model_path, use_gpu, 1);
}

void rs_whisper_free(void * p) {
    rs_ctx * h = (rs_ctx *) p;
    if (!h) return;
    pthread_mutex_lock(&h->mu);
    if (h->ctx) { whisper_free(h->ctx); h->ctx = NULL; }
    rs_str_free(&h->full_text);
    free(h->seg_logprob);
    free(h->initial_prompt);
    free(h->suppress_regex);
    pthread_mutex_unlock(&h->mu);
    pthread_mutex_destroy(&h->mu);
    free(h);
}

int          rs_whisper_handle_uses_gpu(void * p) { rs_ctx * h = (rs_ctx *) p; return h ? h->has_gpu : 0; }
const char * rs_whisper_handle_backend(void * p) { rs_ctx * h = (rs_ctx *) p; return h ? h->backend : ""; }
const char * rs_whisper_model_type(void * p)     { rs_ctx * h = (rs_ctx *) p; return (h && h->ctx) ? whisper_model_type_readable(h->ctx) : ""; }
int          rs_whisper_is_multilingual(void * p){ rs_ctx * h = (rs_ctx *) p; return (h && h->ctx) ? whisper_is_multilingual(h->ctx) : 0; }

/* ----------------------------------------------------------------- knobs */

#define RS_SET_I(name) if (strcmp(key, #name) == 0) { h->opt.name = value; r = 0; } else
#define RS_SET_B(name) if (strcmp(key, #name) == 0) { h->opt.name = value ? true : false; r = 0; } else
#define RS_SET_F(name) if (strcmp(key, #name) == 0) { h->opt.name = value; r = 0; } else

int rs_whisper_set_opt_i(void * p, const char * key, int value) {
    rs_ctx * h = (rs_ctx *) p;
    if (!h || !key) return -1;
    int r = -1;
    pthread_mutex_lock(&h->mu);
    RS_SET_B(no_timestamps)
    RS_SET_B(single_segment)
    RS_SET_B(no_context)
    RS_SET_B(suppress_blank)
    RS_SET_B(suppress_nst)
    RS_SET_B(token_timestamps)
    RS_SET_B(print_progress)
    RS_SET_B(print_special)
    RS_SET_B(print_realtime)
    RS_SET_B(print_timestamps)
    RS_SET_B(split_on_word)
    RS_SET_B(detect_language)
    RS_SET_B(debug_mode)
    RS_SET_I(max_tokens)
    RS_SET_I(audio_ctx)
    RS_SET_I(n_max_text_ctx)
    RS_SET_I(offset_ms)
    RS_SET_I(duration_ms)
    RS_SET_I(max_len)
    if (strcmp(key, "best_of") == 0) { h->opt.greedy.best_of = value; r = 0; }
    else { if (r != 0) r = -1; }
    pthread_mutex_unlock(&h->mu);
    if (r) rs_set_err("unknown int option '%s'", key);
    return r;
}

int rs_whisper_set_opt_f(void * p, const char * key, float value) {
    rs_ctx * h = (rs_ctx *) p;
    if (!h || !key) return -1;
    int r = -1;
    pthread_mutex_lock(&h->mu);
    RS_SET_F(temperature)
    RS_SET_F(temperature_inc)
    RS_SET_F(entropy_thold)
    RS_SET_F(logprob_thold)
    RS_SET_F(no_speech_thold)
    RS_SET_F(length_penalty)
    RS_SET_F(max_initial_ts)
    RS_SET_F(thold_pt)
    RS_SET_F(thold_ptsum)
    if (strcmp(key, "patience") == 0) { h->opt.beam_search.patience = value; r = 0; }
    else { if (r != 0) r = -1; }
    pthread_mutex_unlock(&h->mu);
    if (r) rs_set_err("unknown float option '%s'", key);
    return r;
}

static int rs_dup_into(char ** slot, const char * v) {
    free(*slot); *slot = NULL;
    if (v && *v) {
        size_t n = strlen(v) + 1;
        *slot = (char *) malloc(n);
        if (!*slot) return -1;
        memcpy(*slot, v, n);
    }
    return 0;
}

int rs_whisper_set_opt_s(void * p, const char * key, const char * value) {
    rs_ctx * h = (rs_ctx *) p;
    if (!h || !key) return -1;
    int r = -1;
    pthread_mutex_lock(&h->mu);
    if (strcmp(key, "initial_prompt") == 0) {
        r = rs_dup_into(&h->initial_prompt, value);
        h->opt.initial_prompt = h->initial_prompt;
    } else if (strcmp(key, "suppress_regex") == 0) {
        r = rs_dup_into(&h->suppress_regex, value);
        h->opt.suppress_regex = h->suppress_regex;
    }
    pthread_mutex_unlock(&h->mu);
    if (r) rs_set_err("unknown/failed string option '%s'", key);
    return r;
}

void rs_whisper_reset_opts(void * p) {
    rs_ctx * h = (rs_ctx *) p;
    if (!h) return;
    pthread_mutex_lock(&h->mu);
    free(h->initial_prompt);  h->initial_prompt  = NULL;
    free(h->suppress_regex);  h->suppress_regex  = NULL;
    rs_init_opts(h);
    pthread_mutex_unlock(&h->mu);
}

/* ------------------------------------------------------------------- run */

void rs_whisper_request_abort(void * p) {
    rs_ctx * h = (rs_ctx *) p;
    if (h) h->abort_flag = 1;
}

static double rs_now_ms(void) { return (double) ggml_time_us() / 1000.0; }

/* collects per-segment / overall stats; must be called with the lock held */
static void rs_collect(rs_ctx * h) {
    const int ns = whisper_full_n_segments(h->ctx);
    h->n_segments = ns;

    if (ns > h->seg_logprob_cap) {
        float * q = (float *) realloc(h->seg_logprob, (size_t) ns * sizeof(float));
        if (q) { h->seg_logprob = q; h->seg_logprob_cap = ns; }
    }

    const whisper_token eot = whisper_token_eot(h->ctx);

    rs_str_clear(&h->full_text);
    double sum_all = 0.0; int cnt_all = 0;
    h->max_nsp = 0.0f;

    for (int i = 0; i < ns; ++i) {
        const char * t = whisper_full_get_segment_text(h->ctx, i);
        if (t) rs_str_append(&h->full_text, t);

        const float nsp = whisper_full_get_segment_no_speech_prob(h->ctx, i);
        if (i == 0 || nsp > h->max_nsp) h->max_nsp = nsp;

        double sum = 0.0; int cnt = 0;
        const int nt = whisper_full_n_tokens(h->ctx, i);
        for (int j = 0; j < nt; ++j) {
            if (whisper_full_get_token_id(h->ctx, i, j) >= eot) continue; /* special/timestamp */
            float pr = whisper_full_get_token_p(h->ctx, i, j);
            if (pr <= 0.0f) pr = 1e-10f;
            sum += log((double) pr);
            cnt++;
        }
        sum_all += sum; cnt_all += cnt;
        if (i < h->seg_logprob_cap) {
            h->seg_logprob[i] = cnt ? (float) (sum / cnt) : NAN;
        }
    }
    h->avg_logprob = cnt_all ? (float) (sum_all / cnt_all) : NAN;

    const int lid = whisper_full_lang_id(h->ctx);
    const char * ls = (lid >= 0) ? whisper_lang_str(lid) : NULL;
    snprintf(h->lang, sizeof(h->lang), "%s", ls ? ls : "");

}

/* fires right before the first encoder run -> measures prep/warm-up cost */
static bool rs_encoder_begin_cb(struct whisper_context * ctx, struct whisper_state * st, void * user) {
    (void) ctx; (void) st;
    rs_ctx * h = (rs_ctx *) user;
    if (h->prep_ms < 0.0) {
        h->prep_ms = rs_now_ms() - h->t_run_start;
    }
    return h->abort_flag == 0;   /* false aborts before encoding */
}

int rs_whisper_run(void * p, const float * pcm, int n_samples,
                   const char * language, int translate,
                   int beam_size, int n_threads) {
    rs_ctx * h = (rs_ctx *) p;
    g_err[0] = '\0';
    if (!h || !h->ctx) { rs_set_err("null handle"); return -1; }
    if (!pcm || n_samples <= 0) { rs_set_err("empty pcm buffer"); return -2; }

    pthread_mutex_lock(&h->mu);
    h->abort_flag = 0;

    struct whisper_full_params wp = h->opt;

    if (beam_size > 1) {
        wp.strategy = WHISPER_SAMPLING_BEAM_SEARCH;
        wp.beam_search.beam_size = beam_size;
    } else {
        wp.strategy = WHISPER_SAMPLING_GREEDY;
        if (wp.greedy.best_of <= 0) wp.greedy.best_of = 1;
    }

    wp.translate = translate ? true : false;

    /* keep a stable copy of the language string for the duration of the call */
    char langbuf[32];
    if (language && *language && strcmp(language, "auto") != 0) {
        snprintf(langbuf, sizeof(langbuf), "%s", language);
        if (whisper_lang_id(langbuf) < 0) {
            pthread_mutex_unlock(&h->mu);
            rs_set_err("unknown language code '%s'", langbuf);
            return -3;
        }
        wp.language        = langbuf;
        wp.detect_language = false;
    } else {
        wp.language        = "auto";
        wp.detect_language = false;  /* whisper_full auto-detects when lang=="auto" */
    }

    if (n_threads > 0) wp.n_threads = n_threads;

    wp.abort_callback           = rs_abort_cb;
    wp.abort_callback_user_data = h;
    wp.initial_prompt           = h->initial_prompt;
    wp.suppress_regex           = h->suppress_regex;

    wp.encoder_begin_callback           = rs_encoder_begin_cb;
    wp.encoder_begin_callback_user_data = h;

    const double t0 = rs_now_ms();
    h->t_run_start  = t0;
    h->prep_ms      = -1.0;
    int rc = -100;
#ifdef __cplusplus
    try {
#endif
        rc = whisper_full(h->ctx, wp, pcm, n_samples);
#ifdef __cplusplus
    } catch (const std::bad_alloc &) {
        rs_set_err("out of memory during inference");
        rc = -101;
    } catch (const std::exception & e) {
        rs_set_err("exception during inference: %s", e.what());
        rc = -102;
    } catch (...) {
        rs_set_err("unknown exception during inference");
        rc = -103;
    }
#endif
    h->run_ms   = rs_now_ms() - t0;
    h->audio_ms = 1000.0 * (double) n_samples / (double) WHISPER_SAMPLE_RATE;

    if (rc != 0) {
        h->n_segments = 0;
        rs_str_clear(&h->full_text);
        h->avg_logprob = NAN;
        if (!g_err[0]) {
            rs_set_err(h->abort_flag ? "inference aborted by caller"
                                     : "whisper_full failed (rc=%d)", rc);
        }
        pthread_mutex_unlock(&h->mu);
        return rc > 0 ? -rc : rc;
    }

#ifdef __cplusplus
    try {
#endif
        rs_collect(h);
#ifdef __cplusplus
    } catch (...) {
        rs_set_err("exception while collecting results");
        pthread_mutex_unlock(&h->mu);
        return -104;
    }
#endif

    pthread_mutex_unlock(&h->mu);
    return 0;
}

/* --------------------------------------------------------------- results */

#define RS_GUARD(h, ret) rs_ctx * h = (rs_ctx *) p; if (!h || !h->ctx) return ret;

int rs_whisper_n_segments(void * p) { RS_GUARD(h, 0) return h->n_segments; }

const char * rs_whisper_segment_text(void * p, int i) {
    RS_GUARD(h, "")
    if (i < 0 || i >= h->n_segments) return "";
    const char * t = whisper_full_get_segment_text(h->ctx, i);
    return t ? t : "";
}

int64_t rs_whisper_segment_t0(void * p, int i) {
    RS_GUARD(h, 0)
    if (i < 0 || i >= h->n_segments) return 0;
    return whisper_full_get_segment_t0(h->ctx, i);
}

int64_t rs_whisper_segment_t1(void * p, int i) {
    RS_GUARD(h, 0)
    if (i < 0 || i >= h->n_segments) return 0;
    return whisper_full_get_segment_t1(h->ctx, i);
}

float rs_whisper_segment_no_speech_prob(void * p, int i) {
    RS_GUARD(h, NAN)
    if (i < 0 || i >= h->n_segments) return NAN;
    return whisper_full_get_segment_no_speech_prob(h->ctx, i);
}

float rs_whisper_segment_avg_logprob(void * p, int i) {
    RS_GUARD(h, NAN)
    if (i < 0 || i >= h->n_segments || i >= h->seg_logprob_cap || !h->seg_logprob) return NAN;
    return h->seg_logprob[i];
}

int rs_whisper_segment_n_tokens(void * p, int i) {
    RS_GUARD(h, 0)
    if (i < 0 || i >= h->n_segments) return 0;
    return whisper_full_n_tokens(h->ctx, i);
}

const char * rs_whisper_full_text(void * p)        { RS_GUARD(h, "") return h->full_text.p ? h->full_text.p : ""; }
float        rs_whisper_avg_logprob(void * p)      { RS_GUARD(h, NAN) return h->avg_logprob; }
float        rs_whisper_max_no_speech_prob(void * p){ RS_GUARD(h, 0.0f) return h->n_segments ? h->max_nsp : 0.0f; }
const char * rs_whisper_detected_language(void * p){ RS_GUARD(h, "") return h->lang; }
double       rs_whisper_last_run_ms(void * p)      { RS_GUARD(h, 0.0) return h->run_ms; }
double       rs_whisper_last_audio_ms(void * p)    { RS_GUARD(h, 0.0) return h->audio_ms; }
double       rs_whisper_last_rtf(void * p)         { RS_GUARD(h, 0.0) return h->audio_ms > 0.0 ? h->run_ms / h->audio_ms : 0.0; }
double       rs_whisper_last_prep_ms(void * p)     { RS_GUARD(h, 0.0) return h->prep_ms < 0.0 ? 0.0 : h->prep_ms; }
