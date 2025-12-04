from tensorboard.backend.event_processing import event_accumulator

log_dir = r"D:\dont_move\git_save\all_autoTest\autoTest\rl_models\tensorboard_logs"
ea = event_accumulator.EventAccumulator(log_dir)
ea.Reload()

print("Tags:", ea.Tags())
print("Scalars:", ea.Tags().get('scalars', []))
print("Histograms:", ea.Tags().get('histograms', []))