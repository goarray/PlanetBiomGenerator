# themes
THEMES = {
    "Light Sci-Fi": """
        QMainWindow, QWidget { 
            background-color: #e0e6f0; 
            color: #1a1a2e; 
            font-family: 'Exo 2', sans-serif;
        }
        QGroupBox { 
            border: 1px solid #4a4a8e; 
            border-radius: 8px; 
            margin-top: 10px; 
            font-size: 12px; 
            color: #1a1a2e;
        }
        QGroupBox::title { 
            subcontrol-origin: margin; 
            subcontrol-position: top left; 
            padding: 2px 6px; 
            background-color: #c0c6d0; 
            border-radius: 4px;
        }
        QLabel { 
            color: #1a1a2e; 
            font-size: 12px;
        }
        QCheckBox { 
            color: #1a1a2e; 
            font-size: 12px;
        }
        QCheckBox::indicator { 
            width: 10px; 
            height: 8px; 
            background-color: #c0c6d0; 
            border: 1px solid #4a4a8e;
        }
        QCheckBox::indicator:checked { 
            background-color: #ff6b6b; 
            border: 1px solid #ff6b6b;
        }
        QSlider::groove:horizontal { 
            height: 6px; 
            background: #c0c6d0; 
            border-radius: 4px;
        }
        QSlider::sub-page:horizontal {
            background: #ff6b6b;
            border-radius: 4px;
        }
        QSlider::handle:horizontal { 
            background: #ff6b6b; 
            border: 1px solid #ff6b6b; 
            width: 10px; 
            height: 8px; 
            margin: -6px 0; 
            border-radius: 10px;
        }
        QPushButton { 
            background-color: #c0c6d0; 
            color: #1a1a2e; 
            border: 1px solid #4a4a8e; 
            border-radius: 6px; 
            padding: 8px; 
            font-size: 16px;
        }
        QPushButton:hover { 
            background-color: #d0d6e0; 
            border: 1px solid #ff6b6b;
        }
        QComboBox { 
            background-color: #c0c6d0; 
            color: #1a1a2e; 
            border: 1px solid #4a4a8e; 
            border-radius: 4px; 
            padding: 5px;
        }
        QComboBox QAbstractItemView { 
            background-color: #c0c6d0; 
            color: #1a1a2e; 
            selection-background-color: #d0d6e0;
        }
    """,
    
    "Dark Sci-Fi": """
        QMainWindow, QWidget { 
            background-color: #1a1a2e; 
            color: #e0e0e0; 
            font-family: 'Orbitron', sans-serif;
        }
        QGroupBox { 
            border: 1px solid #4a4a8e; 
            border-radius: 8px; 
            margin-top: 10px; 
            font-size: 12px; 
            color: #00d4ff;
        }
        QGroupBox::title { 
            subcontrol-origin: margin; 
            subcontrol-position: top left; 
            padding: 2px 4px; 
            background-color: #2a2a4e; 
            border-radius: 4px;
        }
        QLabel { 
            color: #b0b0ff; 
            font-size: 12px;
        }
        QCheckBox { 
            color: #e0e0e0; 
            font-size: 12px;
        }
        QCheckBox::indicator { 
            width: 10px; 
            height: 8px; 
            background-color: #2a2a4e; 
            border: 1px solid #4a4a8e;
        }
        QCheckBox::indicator:checked { 
            background-color: #00d4ff; 
            border: 1px solid #00d4ff;
        }
        QSlider::groove:horizontal { 
            height: 6px; 
            background: #2a2a4e; 
            border-radius: 4px;
        }
        QSlider::handle:horizontal { 
            background: #00d4ff; 
            border: 1px solid #00d4ff; 
            width: 10px; 
            height: 8px; 
            margin: -6px 0; 
            border-radius: 10px;
        }
        QSlider::sub-page:horizontal {
            background: #00d4ff;
            border-radius: 4px;
        }
        QPushButton { 
            background-color: #2a2a4e; 
            color: #00d4ff; 
            border: 1px solid #4a4a8e; 
            border-radius: 6px; 
            padding: 8px; 
            font-size: 16px;
        }
        QPushButton:hover { 
            background-color: #3a3a6e; 
            border: 1px solid #00d4ff;
        }
        QComboBox { 
            background-color: #2a2a4e; 
            color: #00d4ff; 
            border: 1px solid #4a4a8e; 
            border-radius: 4px; 
            padding: 5px;
        }
        QComboBox QAbstractItemView { 
            background-color: #2a2a4e; 
            color: #00d4ff; 
            selection-background-color: #3a3a6e;
        }
    """,
}
