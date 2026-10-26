@echo off
cd /d "%~dp0"
echo ========================================================
echo   Pushing AI Image Caption Generator to GitHub
echo   Repository: https://github.com/tejashrikamble365/ai-image-caption-generator
echo ========================================================
echo.
git branch -M main
git push -u origin main
echo.
pause
