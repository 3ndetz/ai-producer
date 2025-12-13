import argparse
from loguru import logger
from src.core.config import settings

def run_web():
    import os
    
    logger.info("\n" + "="*70)
    logger.info("AI Producer - Web Mode")
    logger.info("="*70)
    logger.info(f"\nStarting Chainlit server on {settings.web_host}:{settings.web_port}...")
    logger.info("Press Ctrl+C to stop\n")
    
    logger.info(f"chainlit run src/web/app.py --host {settings.web_host} --port {settings.web_port}")
    os.system(f"chainlit run src/web/app.py --host {settings.web_host} --port {settings.web_port}")


def run_backend():
    import uvicorn
    from src.backend.server import app
    
    logger.info("\n" + "="*70)
    logger.info("AI Producer - Backend Mode")
    logger.info("="*70)
    logger.info(f"Starting backend server on {settings.backend_host}:{settings.backend_port}...")
    logger.info("Press Ctrl+C to stop\n")
    
    uvicorn.run(app, host=settings.backend_host, port=settings.backend_port, log_level="info")


def main():
    parser = argparse.ArgumentParser(
        description="AI Producer - Iterative image generation with agent feedback"
    )
    
    parser.add_argument(
        "--mode",
        choices=["web", "backend"],
        default="web",
        help="Run mode: web (Chainlit interface), backend (FastAPI server)"
    )
    
    args = parser.parse_args()

    
    if args.mode == "web":
        run_web()
    elif args.mode == "backend":
        run_backend()


if __name__ == "__main__":
    main()
