"""
run_app.py

Small launcher script that sets safe defaults and runs the Tkinter app.
Use this as the entry point when packaging into an executable.
"""
import os
import subprocess
import sys

# Optional: set GEMINI_MODEL default for packaging environments
os.environ.setdefault("GEMINI_MODEL", "gemini-2.0-flash")

def main():
    # If this script is bundled (frozen) we should import the app module
    # directly. When running from source, fall back to launching app.py
    if getattr(sys, "frozen", False):
        # Running in a bundle
        try:
            import app

            app_root = getattr(app, "CropApp", None)
            if app_root is None:
                print("app module does not expose CropApp")
                sys.exit(1)
            # instantiate and run the Tk app
            app_instance = app.CropApp()
            app_instance.mainloop()
        except Exception as e:
            print(f"Failed to run embedded app: {e}")
            sys.exit(1)
    else:
        # Launch as a separate process (development mode)
        here = os.path.dirname(__file__)
        app_path = os.path.join(here, "app.py")
        if not os.path.exists(app_path):
            print("app.py not found in project folder")
            sys.exit(1)
        # Run using the same interpreter
        subprocess.run([sys.executable, app_path] + sys.argv[1:])


if __name__ == "__main__":
    main()
