
import numpy as np

def denoise_text(model,tokenizer,noisy_text):

    inputs = tokenizer(noisy_text, return_tensors="pt",max_length=25,truncation=True,padding="max_length")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=30, 
        num_beams=5,
    )
    denoised_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return denoised_text

def systematic(G):
    G = (G.copy() % 2).astype(np.uint8)
    k, n = G.shape
    perm = np.arange(n)
    row = 0
    for col in range(n):
        pivot_rows = np.where(G[row:, col] == 1)[0]
        if pivot_rows.size == 0:
            continue
        pivot = pivot_rows[0] + row
        if pivot != row:
            G[[row, pivot]] = G[[pivot, row]]

        if col != row:
            G[:, [col, row]] = G[:, [row, col]]
            perm[[col, row]] = perm[[row, col]]

        for r in range(k):
            if r != row and G[r, row] == 1:
                G[r, :] ^= G[row, :]

        row += 1
        if row == k:
            break

    return G.astype(np.uint8), perm