import argparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from .data import load_frame, stratified_split


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--data", required=True)
    parser.add_argument("--model", choices=["logistic_regression", "svm"], required=True)
    parser.add_argument("--seed", type=int, default=42); args = parser.parse_args()
    frame = load_frame(args.data); train, _, test = stratified_split(frame, args.seed)
    x_train = ("[TEACHER] " + train.teacher_text + " [STUDENT] " + train.student_text).tolist()
    x_test = ("[TEACHER] " + test.teacher_text + " [STUDENT] " + test.student_text).tolist()
    classifier = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed) if args.model == "logistic_regression" else LinearSVC(class_weight="balanced", random_state=args.seed)
    pipeline = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000), classifier)
    pipeline.fit(x_train, train.label); prediction = pipeline.predict(x_test)
    p, r, f1, _ = precision_recall_fscore_support(test.label, prediction, average="binary", zero_division=0)
    print({"model": args.model, "precision": float(p), "recall": float(r), "f1": float(f1)})


if __name__ == "__main__": main()

