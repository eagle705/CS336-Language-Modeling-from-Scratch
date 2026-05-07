# ML Popup Quiz

macOS menu bar quiz app for reviewing `implementation-practice` concepts in short bursts.

Questions and choices are written in Korean and English together, so the same deck can be used for concept recall and English-answer practice. Some cards ask you to choose the correct code snippet, which is closer to the way these ideas show up in implementation work.

## Run During Development

```bash
cd ml-popup-quiz
swift run
```

## Build App Bundle

```bash
cd ml-popup-quiz
chmod +x build.sh
./build.sh
open ".build/ML Popup Quiz.app"
```

## Add Cards

Edit `Sources/Resources/quiz_bank.json`.

Each card is plain JSON:

```json
{
  "id": "tp-column-row",
  "topic": "Tensor Parallelism",
  "source": "implementation-practice/02-mlp-parallelism/tensor_parallelism.py",
  "difficulty": "core",
  "prompt": "ColumnParallelLinear에서 통신이 필요 없는 forward 구간은?",
  "choices": ["FC1 output shard 계산", "FC2 output sum", "DP gradient sync"],
  "answer": "FC1 output shard 계산",
  "details": "FC1은 hidden/output dimension을 rank별로 나눠 각 rank가 자기 shard만 계산한다.",
  "tags": ["parallelism", "tp", "megatron"]
}
```

Valid `difficulty` values: `warmup`, `core`, `deep`.

## Export To Excel

Click the export icon in the menu bar popup footer, or open Settings and click `Export Excel CSV`.

The exported file is a UTF-8 BOM CSV so Excel opens Korean text correctly.
