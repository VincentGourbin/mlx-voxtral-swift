#!/bin/zsh
# K-89 — runs on the Mac the tests that CI skips because they are guarded by a variable (TEST_RUNNER_VOXTRAL_*:
# real models, long audio, GPU). One `xcodebuild test` per guard, one after the other, never in parallel (CLAUDE.md).
#
#   Scripts/run-guarded-tests.sh            regression guards (default)
#   Scripts/run-guarded-tests.sh --all      + diagnostic probes and campaigns (long; some need extra inputs)
#   Scripts/run-guarded-tests.sh FLAG…      only those guards (e.g. TTS_STREAM CANCEL)
#
# Path guards default to the local model cache: VOXTRAL_K8_PACK_DIR (MarkusKaemmerer 8-bit dense-encoder pack),
# VOXTRAL_RT_ORIGINAL_DIR (mistralai/Voxtral-Mini-4B-Realtime-2602). Logs: .local-runs/guarded/<flag>.log.
# Prints one `GUARDED` line per guard; exit status 1 if any guard has a failure or did not build.
set -u
cd "$(dirname "$0")/.."
OUT=.local-runs/guarded; mkdir -p $OUT
CACHE=$HOME/Library/Caches/models
: ${VOXTRAL_K8_PACK_DIR:=$CACHE/MarkusKaemmerer/Voxtral-Mini-3B-2507-8bit-dense-encoder}
: ${VOXTRAL_RT_ORIGINAL_DIR:=$CACHE/mistralai/Voxtral-Mini-4B-Realtime-2602}

# guard | value | tier | test classes (VoxtralCoreTests) — the 50 tests CI skipped on 2026-10-20 (run 39397564658)
GUARDS=(
  "CANCEL|1|regression|CancellationTests"
  "COREML|1|regression|CoreMLEncoderPathTests"
  "CROSS_ACTOR|1|regression|CrossActorWaveformTests"
  "ENROLL_EXCLUSION|1|regression|EnrollInferenceExclusionTests"
  "ENROLL_REPRO|1|regression|EnrollmentReproTests"
  "K8_PACK_DIR|$VOXTRAL_K8_PACK_DIR|regression|QuantizedThirdPartyPackTests"
  "LONG_AUDIO|1|regression|LongPromptKVCacheTests MaxTokensForDurationTests"
  "PARITY|1|regression|STTMaskParityTests"
  "RT_ORIGINAL_DIR|$VOXTRAL_RT_ORIGINAL_DIR|regression|RealtimeOriginalRepoTests"
  "TTS_CARRIER|1|regression|TTSWarmUpCarrierTrimTests"
  "TTS_CARRIER_EARLY|1|regression|TTSCarrierEarlyCutTests"
  "TTS_CLONE_OVERRUN|1|regression|CloneOverrunTests"
  "TTS_CODEC_BANDED|1|regression|CodecBandedAttentionTests"
  "TTS_PARITY|1|regression|FlowMatchingTeacherForcedParityTests VoicePathParityTests"
  "TTS_REPRO|1|regression|TTSBlindValidationTests TTSConsecutiveSynthesisReproTests TTSGenerateStabilityReproTests TTSReEnrollExperimentTests"
  "TTS_STREAM|1|regression|IncrementalCodecDecodeTests TTSStreamingCancellationTests TTSStreamingPrefixBoundaryTests TTSStreamingTruncationTests"
  "TTS_STREAM_FINAL|1|regression|TTSStreamingFinalChunkTests"
  "TTS_STREAM_SEED|1|regression|TTSStreamingSeedReproTests"
  "UNLOAD_FOOTPRINT|1|regression|UnloadFootprintTests"
  "TTS_CAMPAIGN|1|probe|TTSQuantizationCampaignTests"
  "TTS_COMPARE|1|probe|TTSEmbeddingComparisonTests"
  "TTS_DEFICIT|1|probe|TTSFundamentalDeficitProbeTests"
  "TTS_PROBE|1|probe|TTSCarrierCodeProbeTests"
  "TTS_TWOPASS|1|probe|TTSTwoPassWarmUpProbeTests"
)

ALL=0; ONLY=()
for a in "$@"; do [[ $a == --all ]] && ALL=1 || ONLY+=($a); done

rc=0
for entry in $GUARDS; do
  IFS='|' read -r flag value tier classes <<< "$entry"
  if (( ${#ONLY} )); then (( ${ONLY[(Ie)$flag]} )) || continue
  elif [[ $tier == probe && $ALL == 0 ]]; then continue; fi
  only=(); for c in ${=classes}; do only+=(-only-testing:VoxtralCoreTests/$c); done
  log=$OUT/$flag.log
  env "TEST_RUNNER_VOXTRAL_$flag=$value" xcodebuild test -scheme MLXVoxtralSwift-Package -destination 'platform=macOS' \
    -derivedDataPath .build/xcode-test -parallel-testing-enabled NO -onlyUsePackageVersionsFromResolvedFile \
    $only > $log 2>&1
  code=$?
  summary=$(grep -E '^\s*Executed [0-9]+ test' $log | tail -1 | sed -E 's/^[[:space:]]*//')
  skipped=$(grep -c "' skipped (" $log)
  echo "GUARDED $flag ($classes) : ${summary:-no test run} ; skipped $skipped ; xcodebuild exit $code"
  (( code == 0 )) || rc=1
done
exit $rc
