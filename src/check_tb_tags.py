# src/check_tb_tags.py
import os
from tensorboard.backend.event_processing import event_accumulator

log_dir = r"D:\kannada_ocr_project\logs\20251111-054945\val"  # change if needed

ea = event_accumulator.EventAccumulator(log_dir)
ea.Reload()

print("\n📂 Tags found in training log:")
print(ea.Tags()['scalars'])

if len(ea.Tags()['scalars']) == 0:
    print("⚠️ No scalar tags found in this log. Try opening in TensorBoard manually to verify.")
