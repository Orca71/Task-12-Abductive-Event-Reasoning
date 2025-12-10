from aer.data_loader import load_docs, load_questions, attach_docs

print("=== TRAIN DATA ===")
train_q = load_questions("train_data/questions.jsonl")
train_d = load_docs("train_data/docs.json")
train_q = attach_docs(train_q, train_d)

print("Train examples:", len(train_q))
print("Example topic:", train_q[0].topic_text)
print("Num snippets:", len(train_q[0].docs))
print("First snippet:", train_q[0].docs[0])

print("\n=== DEV DATA ===")
dev_q = load_questions("dev_data/questions.jsonl")
dev_d = load_docs("dev_data/docs.json")
dev_q = attach_docs(dev_q, dev_d)

print("Dev examples:", len(dev_q))
print("Example topic:", dev_q[0].topic_text)
print("Num snippets:", len(dev_q[0].docs))
print("First snippet:", dev_q[0].docs[0])
