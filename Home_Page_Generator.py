import streamlit as st
import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path
from collections import defaultdict
import importlib.util

# === CONFIG ===
PAGES_DIR = "pages"
PAGE_TITLE = "Personal Portfolio Management Tool"

# === PATH RESOLUTION WORKAROUND ===
def add_pages_to_path():
    """Ensure Streamlit can find pages by adding to Python path"""
    pages_path = str(Path(os.getcwd()) / PAGES_DIR)
    if pages_path not in sys.path:
        sys.path.insert(0, pages_path)

add_pages_to_path()

# === PAGE SETUP ===
st.set_page_config(
    page_title=PAGE_TITLE, 
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        padding: 1rem 0;
        border-bottom: 2px solid #e0e0e0;
        margin-bottom: 2rem;
    }
    .tool-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .launch-button {
        margin: 0.2rem;
    }
    .status-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.5rem;
        border-radius: 5px;
    }
    .status-error {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 0.5rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

def get_pages():
    """Get all Python files in pages directory with enhanced metadata"""
    pages = []
    base_dir = Path(os.getcwd())
    
    if not (base_dir / PAGES_DIR).exists():
        return pages
    
    for py_file in (base_dir / PAGES_DIR).rglob("*.py"):
        if py_file.name.startswith(("_", ".")) or "__pycache__" in str(py_file):
            continue
        
        try:
            rel_path = str(py_file.relative_to(base_dir)).replace("\\", "/")
            path_parts = py_file.relative_to(base_dir / PAGES_DIR).parts
            
            if len(path_parts) > 1:
                category = " / ".join(path_parts[:-1])
            else:
                category = "Main Tools"
            
            display_name = py_file.stem.replace("_", " ").title()
            
            pages.append({
                "name": display_name,
                "filename": py_file.name,
                "path": rel_path,
                "abs_path": str(py_file),
                "file": py_file,
                "category": category,
                "exists": py_file.exists(),
                "size": py_file.stat().st_size if py_file.exists() else 0
            })
        except Exception as e:
            st.sidebar.error(f"Error processing {py_file.name}: {str(e)}")
            continue
    
    return sorted(pages, key=lambda x: (x["category"], x["name"]))

def launch_streamlit_app(file_path, port=None):
    """Launch a Streamlit app in a new process"""
    try:
        # Find available port if not specified
        if port is None:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', 0))
                port = s.getsockname()[1]
        
        # Prepare command
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            file_path, "--server.port", str(port),
            "--server.headless", "true"
        ]
        
        # Launch process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.getcwd()
        )
        
        # Wait a moment for the server to start
        time.sleep(2)
        
        # Open browser
        url = f"http://localhost:{port}"
        webbrowser.open(url)
        
        return True, f"App launched at {url}", process
        
    except Exception as e:
        return False, f"Launch failed: {str(e)}", None

def create_launch_buttons(page):
    """Create multiple launch options for each tool"""
    button_key_base = f"launch_{page['name'].replace(' ', '_')}_{page['category'].replace(' ', '_').replace('/', '_')}"
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Method 1: Direct Streamlit page link (for main tools)
        if page["category"] == "Main Tools":
            try:
                st.page_link(f"pages/{page['filename']}", label="🔗 Page Link")
            except:
                st.button("🔗 Page Link", 
                         disabled=True, 
                         help="Not available for this tool",
                         key=f"{button_key_base}_pagelink_disabled")
        else:
            st.button("🔗 Page Link", 
                     disabled=True, 
                     help="Not available for nested tools",
                     key=f"{button_key_base}_pagelink_nested")
    
    with col2:
        # Method 2: Launch in new browser tab
        if st.button("🚀 New Tab", key=f"{button_key_base}_tab"):
            with st.spinner("Launching app..."):
                success, message, process = launch_streamlit_app(page["abs_path"])
                if success:
                    st.success(message)
                    # Store process info in session state for later cleanup
                    if 'launched_processes' not in st.session_state:
                        st.session_state.launched_processes = []
                    st.session_state.launched_processes.append({
                        'name': page['name'],
                        'process': process,
                        'url': message.split(' at ')[1] if ' at ' in message else ''
                    })
                else:
                    st.error(message)
    
    with col3:
        # Method 3: Import and run inline (for compatible tools)
        if st.button("📱 Inline", key=f"{button_key_base}_inline"):
            try:
                # This is experimental - try to import and run the module
                st.info("⚠️ Experimental: Attempting inline execution...")
                
                # Import the module dynamically
                spec = importlib.util.spec_from_file_location(
                    page['name'].replace(' ', '_'), 
                    page['abs_path']
                )
                module = importlib.util.module_from_spec(spec)
                
                # Execute in a try-catch to handle any issues
                try:
                    spec.loader.exec_module(module)
                    st.success(f"✅ {page['name']} loaded inline!")
                except Exception as inline_error:
                    st.error(f"Inline execution failed: {str(inline_error)}")
                    st.info("💡 Try the 'New Tab' option instead")
                    
            except Exception as e:
                st.error(f"Could not load module: {str(e)}")
    
    with col4:
        # Method 4: Copy command to clipboard
        if st.button("📋 Copy Cmd", key=f"{button_key_base}_copy"):
            command = f"streamlit run {page['path']}"
            # Use JavaScript to copy to clipboard
            st.components.v1.html(f"""
                <script>
                navigator.clipboard.writeText('{command}').then(function() {{
                    console.log('Command copied to clipboard');
                }});
                </script>
                <div style="background: #e8f5e8; padding: 10px; border-radius: 5px; margin: 10px 0;">
                    ✅ Copied to clipboard: <code>{command}</code>
                </div>
            """, height=60)

def display_tool_card(page):
    """Display a single tool as a card with launch options"""
    with st.container():
        st.markdown(f"""
        <div class="tool-card">
            <h4>🔧 {page['name']}</h4>
            <p style="margin: 0; color: #666;">📁 {page['path']}</p>
            <p style="margin: 0; color: #666;">📊 {page['size']} bytes</p>
        </div>
        """, unsafe_allow_html=True)
        
        if page['exists']:
            create_launch_buttons(page)
        else:
            st.error("❌ File not found!")

def display_pages():
    """Display all pages organized by category"""
    pages = get_pages()
    
    if not pages:
        st.error(f"❌ No Python files found in the `{PAGES_DIR}` directory!")
        return
    
    # Display statistics
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
        <h3 style="margin: 0; text-align: center;">📊 Portfolio Management Dashboard</h3>
        <div style="display: flex; justify-content: space-around; margin-top: 1rem;">
            <div style="text-align: center;">
                <h2 style="margin: 0;">{len(pages)}</h2>
                <p style="margin: 0;">Total Tools</p>
            </div>
            <div style="text-align: center;">
                <h2 style="margin: 0;">{len(set(p['category'] for p in pages))}</h2>
                <p style="margin: 0;">Categories</p>
            </div>
            <div style="text-align: center;">
                <h2 style="margin: 0;">{sum(1 for p in pages if p['exists'])}</h2>
                <p style="margin: 0;">Available</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Group and display pages
    categories = defaultdict(list)
    for page in pages:
        categories[page["category"]].append(page)
    
    for category, page_list in categories.items():
        category_icon = "🏠" if category == "Main Tools" else "📁"
        st.markdown(f"## {category_icon} {category}")
        
        for page in page_list:
            display_tool_card(page)
            st.markdown("---")

def display_sidebar():
    """Enhanced sidebar with system information and process management"""
    st.sidebar.markdown("# 🎛️ Control Panel")
    
    # Launch method explanation
    st.sidebar.markdown("## 🚀 Launch Methods")
    st.sidebar.markdown("""
    - **🔗 Page Link**: Direct navigation (main tools only)
    - **🚀 New Tab**: Launch in separate browser tab
    - **📱 Inline**: Run within this page (experimental)
    - **📋 Copy Cmd**: Copy terminal command
    """)
    
    # Active processes management
    if 'launched_processes' in st.session_state and st.session_state.launched_processes:
        st.sidebar.markdown("## 🔄 Active Apps")
        for i, proc_info in enumerate(st.session_state.launched_processes):
            col1, col2 = st.sidebar.columns([2, 1])
            with col1:
                st.sidebar.text(proc_info['name'])
            with col2:
                if st.sidebar.button("❌", key=f"kill_{i}", help="Stop app"):
                    try:
                        proc_info['process'].terminate()
                        st.session_state.launched_processes.pop(i)
                        st.rerun()
                    except:
                        pass
    
    # System status
    pages = get_pages()
    st.sidebar.markdown("## 📊 System Status")
    available_pages = sum(1 for p in pages if p['exists'])
    
    if available_pages == len(pages) and len(pages) > 0:
        st.sidebar.success(f"✅ All {len(pages)} tools ready")
    elif available_pages > 0:
        st.sidebar.warning(f"⚠️ {available_pages}/{len(pages)} tools available")
    else:
        st.sidebar.error("❌ No tools found")
    
    # Directory info
    st.sidebar.markdown("## 📁 Directory Info")
    st.sidebar.code(f"Working Dir:\n{os.getcwd()}")
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh", use_container_width=True):
        st.rerun()

def main():
    """Main application function"""
    
    # Page header
    st.markdown(f"<h1 class='main-header'>{PAGE_TITLE}</h1>", unsafe_allow_html=True)
    
    # Welcome message with launch instructions
    st.markdown("""
    ## 👋 Welcome to Your Portfolio Management Suite
    
    This dashboard provides **multiple ways** to launch your tools:
    
    🔗 **Page Link** - Direct navigation (works for main-level tools)  
    🚀 **New Tab** - Opens tool in a new browser tab with its own port  
    📱 **Inline** - Experimental: Run tool within this page  
    📋 **Copy Cmd** - Copy terminal command to clipboard  
    """)
    
    # Important note about new tab launches
    st.info("""
    💡 **New Tab Launch**: This will start the tool on a different port and open it in a new browser tab. 
    Each tool runs independently. Use the sidebar to manage active apps.
    """)
    
    # Main content
    display_pages()
    
    # Display sidebar
    display_sidebar()

# === APPLICATION ENTRY POINT ===
if __name__ == "__main__":
    main()
else:
    main()