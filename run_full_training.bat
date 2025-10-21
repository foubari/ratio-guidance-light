@echo off
REM Complete training script for ratio-guidance-light (Windows)
REM This script runs the full training pipeline with reasonable defaults

echo ======================================
echo Ratio Guidance Training Pipeline
echo ======================================
echo.

REM Check dependencies
echo Checking dependencies...
python -c "import torch; import torchvision; import tqdm; import matplotlib; print('✓ All dependencies found')" 2>nul
if errorlevel 1 (
    echo ✗ Missing dependencies. Installing...
    pip install -r requirements.txt
)

echo.
echo ======================================
echo Step 1/5: Train Standard MNIST DDPM
echo ======================================
python src/train_diffusion.py --dataset standard --epochs 50 --batch_size 128 --lr 1e-4 --num_workers 4 --device cuda
if errorlevel 1 goto error

echo.
echo ======================================
echo Step 2/5: Train Rotated MNIST DDPM
echo ======================================
python src/train_diffusion.py --dataset rotated --epochs 50 --batch_size 128 --lr 1e-4 --num_workers 4 --device cuda
if errorlevel 1 goto error

echo.
echo ======================================
echo Step 3/5: Train Ratio Estimator (Discriminator)
echo ======================================
python src/train_ratio.py --loss_type disc --epochs 30 --batch_size 128 --lr 1e-4 --real_fake_ratio 0.5 --num_workers 4 --device cuda
if errorlevel 1 goto error

echo.
echo ======================================
echo Step 4/5: Unconditional Sampling
echo ======================================
python src/sample.py --dataset standard --num_samples 16 --device cuda
if errorlevel 1 goto error

echo.
echo ======================================
echo Step 5/5: Guided Sampling
echo ======================================
python src/sample.py --dataset standard --num_samples 16 --guided --loss_type disc --guidance_scale 2.0 --condition_dataset rotated --device cuda
if errorlevel 1 goto error

echo.
echo ======================================
echo Training Complete!
echo ======================================
echo.
echo Results:
echo   - Diffusion models: checkpoints/diffusion/{standard,rotated}/
echo   - Ratio model: checkpoints/ratio/disc/
echo   - Samples: outputs/
echo.
echo Try different guidance scales:
echo   python src/sample.py --dataset standard --guided --guidance_scale 0.5
echo   python src/sample.py --dataset standard --guided --guidance_scale 5.0
goto end

:error
echo.
echo ======================================
echo ERROR: Training failed!
echo ======================================
exit /b 1

:end
