/* roenspeak_whisper.h - minimal, ctypes-friendly C ABI over whisper.cpp
 *
 * Design rules (ABI stability for Python ctypes):
 *   - No structs passed or returned by value. Only: void*, int, float, double,
 *     int64_t, const char* (UTF-8, NUL-terminated).
 *   - All returned strings are owned by the library. A segment/full-text pointer
 *     stays valid until the next rs_whisper_run()/rs_whisper_free() on the SAME
 *     handle. Copy it in Python (ctypes c_char_p already copies on .value).
 *   - No C++ exception may cross this boundary (see rs_whisper_run).
 *   - One handle == one whisper_context. rs_whisper_run, the three set_opt
 *     setters, reset_opts and free take that handle's mutex, so concurrent
 *     runs on one handle serialize instead of corrupting state
 *     (whisper_context is not re-entrant). The result accessors listed after
 *     rs_whisper_run do NOT lock, and read whisper_context's internal buffers
 *     directly, so they must be called from the same thread that called
 *     rs_whisper_run, between runs. RoEnSpeak drives one handle from one
 *     dedicated inference thread only.
 *   - rs_whisper_last_error() is thread-local: it reports the last failure on
 *     the calling thread.
 */
#ifndef ROENSPEAK_WHISPER_H
#define ROENSPEAK_WHISPER_H

#include <stdint.h>

#if defined(_WIN32)
#  define RS_API __declspec(dllexport)
#else
#  define RS_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ---- library-level ------------------------------------------------------ */

/* Bump when the meaning/order of exported symbols changes. Currently 2
 * (2 added rs_whisper_set_log_sink). */
#define RS_WHISPER_ABI_VERSION 2
RS_API int          rs_whisper_abi_version(void);
/* whisper.cpp version string, e.g. "1.9.5". */
RS_API const char * rs_whisper_whisper_version(void);
/* git commit the dylib was built from (baked in at compile time, may be ""). */
RS_API const char * rs_whisper_build_commit(void);
/* whisper_print_system_info(): "METAL = 1 | BLAS = 1 | ..." */
RS_API const char * rs_whisper_system_info(void);
/* Comma separated list of ggml devices, e.g.
 * "MTL0:Apple M3 Pro(type=gpu,free=..MiB,total=..MiB), CPU:...". Returns a
 * single static buffer: copy it before the next call (callers are expected to
 * query this from one thread, e.g. the probe). */
RS_API const char * rs_whisper_devices(void);
/* 1 if a ggml GPU device (Metal) is registered in this process, else 0. */
RS_API int          rs_whisper_metal_available(void);

/* 0 = silent, 1 = errors only (default), 2 = everything. Errors are always
 * also captured into rs_whisper_last_error(). Global (whisper_log_set and
 * ggml_log_set are process-wide), so set it once at startup. */
RS_API void         rs_whisper_set_log_level(int level);

/* Where the lines selected by rs_whisper_set_log_level() go. NULL (the default)
 * means stderr; any other value means "call this instead of writing to
 * stderr", which is how a GUI app routes whisper/ggml logging into its own
 * logger without a bundled .app spraying stdout/stderr.
 *
 * level is a ggml_log_level (1=error, 2=warn, 3=info, 4=debug, 5=cont),
 * text is a NUL-terminated UTF-8 fragment (NOT necessarily a whole line, and
 * usually ends with '\n'); it is only valid for the duration of the call.
 * The sink may be invoked from any thread ggml/whisper logs on, including
 * while a rs_whisper_run() is in flight, and must not call back into this
 * library. Global; pass NULL to uninstall (do that before unloading the
 * caller's own code). */
typedef void (*rs_whisper_log_fn)(int level, const char * text, void * user);
RS_API void         rs_whisper_set_log_sink(rs_whisper_log_fn fn, void * user);

/* Last error on the CALLING thread, "" if none. Never NULL. */
RS_API const char * rs_whisper_last_error(void);

/* ---- model handle ------------------------------------------------------- */

/* Loads a ggml model. use_gpu != 0 -> Metal. flash_attn != 0 -> flash attention.
 * Returns an opaque handle, or NULL (see rs_whisper_last_error()). */
RS_API void * rs_whisper_init_ex(const char * model_path, int use_gpu, int flash_attn);
/* Convenience: rs_whisper_init_ex(model_path, use_gpu, 1). */
RS_API void * rs_whisper_init(const char * model_path, int use_gpu);
RS_API void   rs_whisper_free(void * h);

/* 1 if this handle was created with use_gpu and a Metal device exists. */
RS_API int          rs_whisper_handle_uses_gpu(void * h);
/* "Metal" or "CPU" - the backend this handle actually got. */
RS_API const char * rs_whisper_handle_backend(void * h);
/* e.g. "base", "large v3" */
RS_API const char * rs_whisper_model_type(void * h);
RS_API int          rs_whisper_is_multilingual(void * h);

/* ---- optional knobs (applied to every subsequent run) ------------------- */
/* Return 0 on success, -1 on unknown key. Keys (int):
 *   no_timestamps, single_segment, no_context, suppress_blank, suppress_nst,
 *   token_timestamps, print_progress, print_special, print_realtime,
 *   print_timestamps, max_tokens, audio_ctx, best_of, n_max_text_ctx,
 *   offset_ms, duration_ms, max_len, split_on_word, detect_language
 * Keys (float):
 *   temperature, temperature_inc, entropy_thold, logprob_thold,
 *   no_speech_thold, length_penalty, max_initial_ts, patience
 * Keys (string, copied):
 *   initial_prompt, suppress_regex  (NULL/"" clears) */
RS_API int rs_whisper_set_opt_i(void * h, const char * key, int   value);
RS_API int rs_whisper_set_opt_f(void * h, const char * key, float value);
RS_API int rs_whisper_set_opt_s(void * h, const char * key, const char * value);
/* Reset all knobs to whisper_full_default_params() + our defaults. */
RS_API void rs_whisper_reset_opts(void * h);

/* ---- inference --------------------------------------------------------- */

/* pcm: float32, 16 kHz, mono, range [-1,1]. Not retained after return.
 * language: "ro", "en", ... or NULL/""/"auto" for auto-detect.
 * translate != 0  -> task=translate (X -> English).
 * beam_size > 1   -> WHISPER_SAMPLING_BEAM_SEARCH with that beam,
 * beam_size <= 1  -> WHISPER_SAMPLING_GREEDY (best_of knob applies).
 * n_threads <= 0  -> auto (min(4, hw_concurrency)).
 * Returns 0 on success, <0 on failure. Never throws. */
RS_API int rs_whisper_run(void * h, const float * pcm, int n_samples,
                          const char * language, int translate,
                          int beam_size, int n_threads);

/* Ask the running inference to stop early (safe to call from another thread).
 * The in-flight rs_whisper_run() returns non-zero. Auto-cleared on next run. */
RS_API void rs_whisper_request_abort(void * h);

/* ---- results of the last successful run -------------------------------- */

RS_API int          rs_whisper_n_segments(void * h);
RS_API const char * rs_whisper_segment_text(void * h, int i);      /* UTF-8, "" on error */
RS_API int64_t      rs_whisper_segment_t0(void * h, int i);        /* centiseconds */
RS_API int64_t      rs_whisper_segment_t1(void * h, int i);
RS_API float        rs_whisper_segment_no_speech_prob(void * h, int i);
/* mean log(p) over the segment's non-special tokens (NaN if none). */
RS_API float        rs_whisper_segment_avg_logprob(void * h, int i);
RS_API int          rs_whisper_segment_n_tokens(void * h, int i);

/* All segment texts concatenated (leading space of each segment kept as-is). */
RS_API const char * rs_whisper_full_text(void * h);
/* mean log(p) over every non-special token of the run (NaN if none). */
RS_API float        rs_whisper_avg_logprob(void * h);
/* max over segments (0.0 if no segments) - useful as a "silence" gate. */
RS_API float        rs_whisper_max_no_speech_prob(void * h);
/* language actually used/detected, e.g. "ro". "" if unknown. */
RS_API const char * rs_whisper_detected_language(void * h);
/* wall-clock ms of the last rs_whisper_run(). */
RS_API double       rs_whisper_last_run_ms(void * h);
/* audio duration fed to the last run, in ms. */
RS_API double       rs_whisper_last_audio_ms(void * h);
/* run_ms / audio_ms (lower is faster than realtime). */
RS_API double       rs_whisper_last_rtf(void * h);
/* ms from entering rs_whisper_run() until the encoder actually started, i.e.
 * resample/mel + graph/Metal pipeline setup. Big on the first call (warm-up). */
RS_API double       rs_whisper_last_prep_ms(void * h);

/* ---- helpers ----------------------------------------------------------- */
/* -1 if unknown. Use to validate a language code before a run. */
RS_API int rs_whisper_lang_id(const char * lang);

#ifdef __cplusplus
}
#endif
#endif /* ROENSPEAK_WHISPER_H */
