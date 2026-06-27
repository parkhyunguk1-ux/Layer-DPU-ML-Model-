"""
MLOps 파이프라인 엔트리포인트
1. 샘플 데이터 생성 (또는 실제 CSV 사용)
2. 모니터링 (DPU 드리프트 감지)
3. 모델 학습
4. API 서버 실행
5. 대시보드 실행
"""

import argparse
import subprocess
import sys


def run(cmd: str):
    print(f"\n▶ {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ 실패: {cmd}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="LCD DPU MLOps 파이프라인")
    parser.add_argument("--step", choices=["all", "data", "monitor", "train", "api", "dashboard"],
                        default="all", help="실행할 단계")
    parser.add_argument("--csv",  default="data.csv", help="CSV 파일 경로")
    parser.add_argument("--port", default=8000, type=int, help="API 포트")
    args = parser.parse_args()

    py = sys.executable

    if args.step in ("all", "data"):
        print("\n[1/4] 샘플 데이터 생성")
        run(f"{py} generate_sample_data.py")

    if args.step in ("all", "monitor"):
        print("\n[2/4] DPU 드리프트 모니터링")
        run(f"{py} src/monitor.py")

    if args.step in ("all", "train"):
        print("\n[3/4] 모델 학습")
        run(f"{py} src/train.py")

    if args.step == "api":
        print(f"\n[API 서버] http://localhost:{args.port}/docs")
        run(f"uvicorn src.api:app --host 0.0.0.0 --port {args.port} --reload")

    if args.step == "dashboard":
        print("\n[대시보드] http://localhost:8501")
        run(f"streamlit run dashboard/app.py")

    if args.step == "all":
        print("\n✅ 파이프라인 완료!")
        print("   API 실행:       python pipeline.py --step api")
        print("   대시보드 실행:  python pipeline.py --step dashboard")


if __name__ == "__main__":
    main()
