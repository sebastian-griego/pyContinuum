@echo off
echo Uploading pycontinuum to PyPI...
echo.
echo Make sure you have your PyPI token ready (starts with pypi-...)
echo.
set /p TOKEN="Paste your PyPI token here: "
echo.
python -m twine upload dist/* -u __token__ -p %TOKEN%
echo.
echo Done!
pause