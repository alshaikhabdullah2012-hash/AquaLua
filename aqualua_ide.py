import tkinter as tk
from tkinter import font, filedialog, messagebox
import re
import os
import subprocess
import threading
import tempfile

# ---------------- COLORS ----------------
BG = "#0b0e14"
TOP = "#0d111c"
EDITOR = "#0f1422"
ACCENT = "#7aa2f7"
TEXT = "#c0caf5"
KEYWORD = "#ff9e64"
BORDER = "#2a2f45"
TERM_BG = "#000000"
TERM_FG = "#7aff7a"

# ---------------- KEYWORDS ----------------
KEYWORDS = {
    "fn", "return", "if", "else", "while", "for",
    "break", "continue", "true", "false", "null",
    "class", "import","const", "let", "var", "static",
    "public", "private", "protected", "void", "int", "float",
    "string", "bool", "new", "this", "super", "switch", "case", "default",
    "try", "catch", "finally", "throw", "extends", "implements",
    "print", "input","int","float","string","bool","ast_exec"
}

# ---------------- GLOBAL VARS ----------------
current_file = None

# ---------------- WINDOW ----------------
root = tk.Tk()
root.title("AquaLua IDE")
root.configure(bg=BG)
root.attributes("-fullscreen", True)

# ---------------- LOGO ----------------
def setup_logo():
    try:
        logo_path = r"C:\Users\abood\Downloads\Aqualua\AquaLua logo\AquaLua logo.png"
        if os.path.exists(logo_path):
            root.iconbitmap(default=logo_path)
        else:
            # Try relative path
            logo_path = "AquaLua logo.png"
            if os.path.exists(logo_path):
                root.iconbitmap(default=logo_path)
    except:
        pass  # Logo not found, continue without it

setup_logo()

# ---------------- FUNCTIONS ----------------
def new_file():
    global current_file
    editor.delete('1.0', tk.END)
    current_file = None
    root.title("AquaLua IDE - New File")

def open_file():
    global current_file
    file_path = filedialog.askopenfilename(
        title="Open AquaLua File",
        filetypes=[("AquaLua files", "*.aq"), ("All files", "*.*")]
    )
    if file_path:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            editor.delete('1.0', tk.END)
            editor.insert('1.0', content)
            current_file = file_path
            root.title(f"AquaLua IDE - {os.path.basename(file_path)}")
            highlight_keywords()
        except Exception as e:
            messagebox.showerror("Error", f"Could not open file: {e}")

def save_file():
    global current_file
    if current_file:
        try:
            with open(current_file, 'w', encoding='utf-8') as f:
                f.write(editor.get('1.0', tk.END + '-1c'))
            messagebox.showinfo("Saved", f"File saved: {os.path.basename(current_file)}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save file: {e}")
    else:
        save_as_file()

def save_as_file():
    global current_file
    file_path = filedialog.asksaveasfilename(
        title="Save AquaLua File",
        defaultextension=".aq",
        filetypes=[("AquaLua files", "*.aq"), ("All files", "*.*")]
    )
    if file_path:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(editor.get('1.0', tk.END + '-1c'))
            current_file = file_path
            root.title(f"AquaLua IDE - {os.path.basename(file_path)}")
            messagebox.showinfo("Saved", f"File saved: {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save file: {e}")

def run_file():
    # Create temp file if no file saved
    if not current_file:
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.aq', delete=False, encoding='utf-8')
        temp_file.write(editor.get('1.0', tk.END + '-1c'))
        temp_file.close()
        file_to_run = temp_file.name
        display_name = "temp.aq"
    else:
        # Save existing file
        try:
            with open(current_file, 'w', encoding='utf-8') as f:
                f.write(editor.get('1.0', tk.END + '-1c'))
        except Exception as e:
            messagebox.showerror("Error", f"Could not save file: {e}")
            return
        file_to_run = current_file
        display_name = os.path.basename(current_file)
    
    # Clear terminal
    terminal.config(state="normal")
    terminal.delete('1.0', tk.END)
    terminal.insert(tk.END, f"Running {display_name}...\n")
    terminal.config(state="disabled")
    terminal.update()
    
    # Run in thread
    def run_thread():
        try:
            # Find CLI in multiple locations
            current_dir = os.path.dirname(os.path.abspath(__file__))
            possible_paths = [
                os.path.join(current_dir, 'aqualua_cli.py'),  # Same directory
                os.path.join(current_dir, '..', 'aqualua_cli.py'),  # Parent directory
                os.path.join(os.path.dirname(current_dir), 'aqualua_cli.py'),  # Root
            ]
            
            cli_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    cli_path = path
                    break
            
            if not cli_path:
                root.after(0, lambda: show_terminal_error(f"CLI not found in: {possible_paths}"))
                return
            
            # Add debug info
            root.after(0, lambda: debug_terminal(f"CLI path: {cli_path}\nFile: {file_to_run}\n"))
            
            # Set environment to prevent CLI from waiting for input
            env = os.environ.copy()
            env['AQUALUA_IDE_RUN'] = '1'
            
            result = subprocess.run(
                ['python', cli_path, file_to_run],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=os.path.dirname(cli_path),
                env=env,
                shell=True
            )
            
            # Clean up temp file
            if not current_file:
                try:
                    os.unlink(file_to_run)
                except:
                    pass
            
            root.after(0, lambda: update_terminal(result))
            
        except subprocess.TimeoutExpired:
            root.after(0, lambda: show_terminal_error("Program timed out after 10 seconds"))
        except FileNotFoundError:
            root.after(0, lambda: show_terminal_error("Python not found in PATH"))
        except Exception as e:
            root.after(0, lambda: show_terminal_error(f"Execution error: {str(e)}"))
    
    threading.Thread(target=run_thread, daemon=True).start()

def debug_terminal(msg):
    terminal.config(state="normal")
    terminal.insert(tk.END, msg)
    terminal.config(state="disabled")
    terminal.update()

def update_terminal(result):
    terminal.config(state="normal")
    
    if result.stdout:
        terminal.insert(tk.END, result.stdout)
    
    if result.stderr:
        terminal.insert(tk.END, f"\nErrors:\n{result.stderr}")
    
    if result.returncode == 0:
        terminal.insert(tk.END, "\n✅ Completed successfully!")
    else:
        terminal.insert(tk.END, f"\n❌ Failed (exit code: {result.returncode})")
    
    terminal.config(state="disabled")
    terminal.see(tk.END)

def show_terminal_error(error):
    terminal.config(state="normal")
    terminal.insert(tk.END, f"\n❌ Error: {error}")
    terminal.config(state="disabled")

# ---------------- TOOLBAR ----------------
toolbar = tk.Frame(root, bg=TOP, height=48)
toolbar.pack(fill="x")

def make_button(text, command=None):
    btn = tk.Label(
        toolbar,
        text=text,
        bg="#151b2f",
        fg=TEXT,
        padx=12,
        pady=6,
        font=("Segoe UI", 10),
        cursor="hand2"
    )
    btn.pack(side="left", padx=6, pady=6)

    def on_enter(e):
        btn.config(bg=ACCENT, fg="black")

    def on_leave(e):
        btn.config(bg="#151b2f", fg=TEXT)
    
    def on_click(e):
        if command:
            command()

    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)
    btn.bind("<Button-1>", on_click)
    return btn

# Create buttons with functions
make_button("New", new_file)
make_button("Open", open_file)
make_button("Save", save_file)
make_button("Run", run_file)
make_button("Format")
make_button("Exit", lambda: root.quit())

# ---------------- EDITOR ----------------
editor_font = font.Font(family="JetBrains Mono", size=13)

editor = tk.Text(
    root,
    bg=EDITOR,
    fg=TEXT,
    insertbackground=ACCENT,
    relief="flat",
    font=editor_font,
    undo=True,
    padx=20,
    pady=20
)
editor.pack(fill="both", expand=True)

editor.insert("1.0", """// AquaLua
fn main() {
    if true {
        return
    }
}""")

# ---------------- TERMINAL ----------------
terminal = tk.Text(
    root,
    height=9,
    bg=TERM_BG,
    fg=TERM_FG,
    relief="flat",
    font=("JetBrains Mono", 12),
    padx=12,
    pady=12
)
terminal.pack(fill="x")
terminal.insert("1.0", "Output will appear here...\n")
terminal.config(state="disabled")

# ---------------- TAGS ----------------
editor.tag_configure("keyword", foreground=KEYWORD)

# ---------------- KEYWORD HIGHLIGHT ----------------
def highlight_keywords(event=None):
    editor.tag_remove("keyword", "1.0", "end")
    text = editor.get("1.0", "end-1c")

    for kw in KEYWORDS:
        for match in re.finditer(rf"\b{kw}\b", text):
            start = f"1.0+{match.start()}c"
            end = f"1.0+{match.end()}c"
            editor.tag_add("keyword", start, end)

editor.bind("<KeyRelease>", highlight_keywords)

# ---------------- AUTO CLOSE ----------------
PAIRS = {"(": ")", "{": "}", "[": "]", "\"": "\"", "'": "'"}

def auto_close(event):
    if event.char in PAIRS:
        editor.insert("insert", PAIRS[event.char])
        editor.mark_set("insert", "insert-1c")

editor.bind("<Key>", auto_close)

# ---------------- AUTO INDENT ----------------
def auto_indent(event):
    line = editor.get("insert linestart", "insert")
    indent = len(line) - len(line.lstrip(" "))
    editor.insert("insert", "\n" + " " * indent)
    return "break"

editor.bind("<Return>", auto_indent)

# ---------------- KEYBINDINGS ----------------
root.bind("<Control-n>", lambda e: new_file())
root.bind("<Control-o>", lambda e: open_file())
root.bind("<Control-s>", lambda e: save_file())
root.bind("<F5>", lambda e: run_file())
root.bind("<Escape>", lambda e: root.attributes("-fullscreen", False))

# ---------------- STARTUP ----------------
editor.delete('1.0', tk.END)
editor.insert('1.0', '''# Welcome to AquaLua IDE!
# Press F5 to run, Ctrl+O to open files

print("Hello, AquaLua!")

let x = 42
let name = "World"

fn greet(name) {
    return "Hello, " + name + "!"
}

print(greet(name))
print("x =", x)
''')

highlight_keywords()
root.mainloop()
