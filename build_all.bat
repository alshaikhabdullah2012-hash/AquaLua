@echo off
echo 🚀 Building Complete AquaLua Distribution...
echo.

echo 📦 Step 1: Installing PyInstaller...
python -m pip install pyinstaller

echo.
echo 🔨 Step 2: Building C Runtime...
python build_runtime.py

echo.
echo 📱 Step 3: Building Executables...
python build_exe.py

echo.
echo 📦 Step 4: Creating Installer Package...
python build_installer.py

echo.
echo ✅ Build Complete!
echo 📁 Check 'AquaLua_Installer' folder for distribution package
echo.
pause