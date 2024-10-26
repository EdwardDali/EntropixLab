from gui import EntropixTGUI
from graphics import integrate_visualization
import logging

logging.basicConfig(level=logging.INFO)

def main():
    # Create main application
    app = EntropixTGUI()
    
    # Initialize visualization
    viz_manager = integrate_visualization(app)
    
    # Start application
    app.root.mainloop()

if __name__ == "__main__":
    main()