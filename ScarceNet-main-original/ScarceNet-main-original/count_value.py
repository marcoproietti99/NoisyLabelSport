import ast

logfile = "selected_samples_log.txt"
outfile = "selected_counts_per_batch.txt"

with open(logfile, "r") as f, open(outfile, "w") as out:
    for i, line in enumerate(f):
        batch_indices = ast.literal_eval(line.strip())
        counts = [len(indices) for indices in batch_indices]
        out.write(f"Batch {i}: selezionati per elemento {counts} (batch size: {len(batch_indices)})\n")