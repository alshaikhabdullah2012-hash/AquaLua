import subprocess
import os
import sys

def build_executables():
    """Build AquaLua CLI and IDE as EXE files"""
    
    print("Building AquaLua Executables...")
    
    # Check if PyInstaller is installed
    try:
        import PyInstaller
    except ImportError:
        print("📦 Installing PyInstaller...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)
    
    logo_path = r"C:\Users\abood\Downloads\Aqualua\AquaLua logo\AquaLua logo.png"
    
    # Build CLI executable
    print("Building AquaLua CLI...")
    cli_cmd = [
        "pyinstaller",
        "--onefile",
        "--name=aqualua",
        f"--icon={logo_path}",
        "--distpath=dist",
        "--workpath=build",
        "--specpath=build",
        "aqualua_cli.py"
    ]
    
    try:
        subprocess.run(cli_cmd, check=True)
        print("CLI built successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ CLI build failed: {e}")
        return False
    
    # Build IDE executable
    print("Building AquaLua IDE...")
    
    # First, update IDE to include logo
    update_ide_with_logo()
    
    ide_cmd = [
        "pyinstaller",
        "--onefile",
        "--windowed",
        "--name=aqualua-ide",
        f"--icon={logo_path}",
        "--distpath=dist",
        "--workpath=build",
        "--specpath=build",
        "aqualua_ide_with_logo.py"
    ]
    
    try:
        subprocess.run(ide_cmd, check=True)
        print("IDE built successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ IDE build failed: {e}")
        return False
    
    print("\nBuild Complete!")
    print("Executables created in 'dist' folder:")
    print("   - aqualua.exe (CLI)")
    print("   - aqualua-ide.exe (IDE)")
    
    return True

def update_ide_with_logo():
    """Create IDE version with logo"""
    
    # Read current IDE
    with open("aqualua_ide.py", "r", encoding="utf-8") as f:
        ide_content = f.read()
    
    # Add logo setup after imports
    logo_code = '''
# ---------------- LOGO ----------------
def setup_logo():
    try:
        logo_path = r"C:\\Users\\abood\\Downloads\\Aqualua\\AquaLua logo\\AquaLua logo.png"
        if os.path.exists(logo_path):
            root.iconbitmap(default=logo_path)
        else:
            # Try relative path
            logo_path = "AquaLua logo.png"
            if os.path.exists(logo_path):
                root.iconbitmap(default=logo_path)
    except:
        pass  # Logo not found, continue without it

'''
    
    # Insert logo code after imports
    import_end = ide_content.find("# ---------------- COLORS ----------------")
    if import_end != -1:
        ide_content = ide_content[:import_end] + logo_code + ide_content[import_end:]
    
    # Add logo setup call after window creation
    window_setup = 'root.attributes("-fullscreen", True)'
    if window_setup in ide_content:
        ide_content = ide_content.replace(
            window_setup,
            window_setup + "\nsetup_logo()"
        )
    
    # Write updated IDE
    with open("aqualua_ide_with_logo.py", "w", encoding="utf-8") as f:
        f.write(ide_content)

if __name__ == "__main__":
    build_executables()