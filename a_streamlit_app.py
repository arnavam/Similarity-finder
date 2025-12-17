"""
Streamlit Frontend for Code Similarity Checker
Communicates with api_server.py via HTTP requests.

Two modes:
- Direct: Upload → Select target → Analyze (one-pass, memory efficient)
- Preprocess: Upload → Preprocess → Select target → Compare (keeps embeddings)
"""

import os

import numpy as np
import requests
import streamlit as st


# API endpoint - tries localhost first, falls back to remote
try:
    requests.get("http://localhost:8000/", timeout=1)
    API_BASE_URL = "http://localhost:8000"
except:
    API_BASE_URL = "https://huggingface.co/spaces/arnavam/copyadi-finder"


# ===== Auth Helper Functions =====


def login_user(username: str, password: str) -> bool:
    """Login and store token."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/auth/login",
            json={"username": username, "password": password},
        )
        if response.status_code == 200:
            data = response.json()
            st.session_state.auth_token = data["access_token"]
            st.session_state.username = data["username"]
            return True
        else:
            st.error(response.json().get("detail", "Login failed"))
            return False
    except Exception as e:
        st.error(f"Login error: {e}")
        return False


def register_user(username: str, password: str, invite_code: str = None) -> bool:
    """Register a new user."""
    try:
        payload = {"username": username, "password": password}
        if invite_code:
            payload["invite_code"] = invite_code
        
        response = requests.post(
            f"{API_BASE_URL}/auth/register",
            json=payload,
        )
        if response.status_code == 200:
            st.success(f"✅ Registered! Please login with your credentials.")
            return True
        else:
            # Handle non-JSON error responses
            try:
                error_detail = response.json().get("detail", "Registration failed")
            except:
                error_detail = f"Registration failed (HTTP {response.status_code})"
            st.error(error_detail)
            return False
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API server. Is it running?")
        return False
    except Exception as e:
        st.error(f"Registration error: {e}")
        return False


def logout_user():
    """Clear auth state."""
    st.session_state.auth_token = None
    st.session_state.username = None


def is_logged_in() -> bool:
    """Check if user is logged in."""
    return st.session_state.get("auth_token") is not None


# ===== API Functions =====


def get_auth_headers() -> dict:
    """Get authorization headers if user is logged in."""
    token = st.session_state.get("auth_token")
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def upload_files_to_api(files, ignore_patterns=None):
    """Upload files to API and get extracted text back."""
    if not is_logged_in():
        st.error("Please login first")
        return {}

    file_data = []
    for f in files:
        f.seek(0)
        file_data.append(("files", (f.name, f.read(), "application/octet-stream")))

    params = {}
    if ignore_patterns:
        params["ignore_patterns"] = ",".join(ignore_patterns)

    try:
        response = requests.post(
            f"{API_BASE_URL}/upload",
            files=file_data,
            params=params,
            headers=get_auth_headers(),
        )
        response.raise_for_status()
        return response.json()["submissions"]
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            st.error("Session expired. Please login again.")
            logout_user()
        else:
            st.error(f"API Error: {e}")
        return {}
    except Exception as e:
        st.error(f"API Error: {e}")
        return {}


def process_github_via_api(urls, ignore_patterns=None):
    """Process GitHub URLs via API."""
    if not is_logged_in():
        st.error("Please login first")
        return {}

    try:
        response = requests.post(
            f"{API_BASE_URL}/github",
            json={"urls": urls, "ignore_patterns": ignore_patterns},
            headers=get_auth_headers(),
        )
        response.raise_for_status()
        return response.json()["submissions"]
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            st.error("Session expired. Please login again.")
            logout_user()
        else:
            st.error(f"API Error: {e}")
        return {}
    except Exception as e:
        st.error(f"API Error: {e}")
        return {}


def fetch_github_history(instance_id=None):
    """Fetch user's GitHub URL history from API."""
    if not is_logged_in():
        return []

    try:
        params = {"instance_id": instance_id} if instance_id else {}
        response = requests.get(
            f"{API_BASE_URL}/github-history",
            params=params,
            headers=get_auth_headers(),
        )
        response.raise_for_status()
        return response.json().get("history", [])
    except:
        return []


def fetch_instances():
    """Fetch user's instances from API."""
    if not is_logged_in():
        return []

    try:
        response = requests.get(
            f"{API_BASE_URL}/instances",
            headers=get_auth_headers(),
        )
        response.raise_for_status()
        return response.json().get("instances", [])
    except:
        return []


def create_new_instance(name, description=""):
    """Create a new instance via API."""
    if not is_logged_in():
        return None

    try:
        response = requests.post(
            f"{API_BASE_URL}/instances",
            params={"name": name, "description": description},
            headers=get_auth_headers(),
        )
        response.raise_for_status()
        return response.json().get("instance")
    except Exception as e:
        st.error(f"Error creating instance: {e}")
        return None


def update_instance_urls_api(instance_id, urls):
    """Update URLs for an instance via API."""
    if not is_logged_in():
        return False

    try:
        response = requests.put(
            f"{API_BASE_URL}/instances/{instance_id}/urls",
            json=urls,
            headers=get_auth_headers(),
        )
        response.raise_for_status()
        return True
    except:
        return False


def delete_instance_api(instance_id):
    """Delete an instance via API."""
    if not is_logged_in():
        return False

    try:
        response = requests.delete(
            f"{API_BASE_URL}/instances/{instance_id}",
            headers=get_auth_headers(),
        )
        response.raise_for_status()
        return True
    except:
        return False


def analyze_direct_via_api(submissions, target_file, preprocessing_options):
    """Direct analysis: parallel process+compare+discard (memory efficient)."""
    if not is_logged_in():
        st.error("Please login first")
        return None

    try:
        response = requests.post(
            f"{API_BASE_URL}/analyze",
            json={
                "submissions": submissions,
                "target_file": target_file,
                "preprocessing_options": preprocessing_options,
                "mode": "direct",
            },
            headers=get_auth_headers(),
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            st.error("Session expired. Please login again.")
            logout_user()
        else:
            st.error(f"API Error: {e}")
        return None
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


def preprocess_via_api(submissions, preprocessing_options):
    """Preprocess all submissions and return embeddings."""
    if not is_logged_in():
        st.error("Please login first")
        return None

    try:
        response = requests.post(
            f"{API_BASE_URL}/preprocess",
            json={
                "submissions": submissions,
                "preprocessing_options": preprocessing_options,
            },
            headers=get_auth_headers(),
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            st.error("Session expired. Please login again.")
            logout_user()
        else:
            st.error(f"API Error: {e}")
        return None
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


def compare_via_api(embeddings, target_file, preprocessing_options):
    """Compare target against preprocessed embeddings."""
    if not is_logged_in():
        st.error("Please login first")
        return None

    try:
        response = requests.post(
            f"{API_BASE_URL}/compare",
            json={
                "embeddings": embeddings,
                "target_file": target_file,
                "preprocessing_options": preprocessing_options,
            },
            headers=get_auth_headers(),
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            st.error("Session expired. Please login again.")
            logout_user()
        else:
            st.error(f"API Error: {e}")
        return None
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


def upload_individual_to_api(files, ignore_patterns=None):
    """Upload files where each file/ZIP = ONE submission."""
    if not is_logged_in():
        st.error("Please login first")
        return {}

    file_data = []
    for f in files:
        f.seek(0)
        file_data.append(("files", (f.name, f.read(), "application/octet-stream")))

    params = {}
    if ignore_patterns:
        params["ignore_patterns"] = ",".join(ignore_patterns)

    try:
        response = requests.post(
            f"{API_BASE_URL}/upload-individual",
            files=file_data,
            params=params,
            headers=get_auth_headers(),
        )
        response.raise_for_status()
        return response.json()["submissions"]
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            st.error("Session expired. Please login again.")
            logout_user()
        else:
            st.error(f"API Error: {e}")
        return {}
    except Exception as e:
        st.error(f"API Error: {e}")
        return {}


def display_results(result):
    """Display similarity analysis results."""
    st.header("📈 Results")

    scores = result["scores"]
    submission_names = result["submission_names"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Submissions", len(submission_names))
    with col2:
        st.metric("Score Types", len(scores))
    with col3:
        if scores:
            all_values = [v for vals in scores.values() for v in vals if vals]
            if all_values:
                max_score = max(all_values)
                st.metric("Highest Similarity", f"{max_score:.2%}")

    st.subheader("Similar Submissions Found")
    for score_type, score_values in scores.items():
        if score_values:
            max_idx = np.argmax(score_values)
            st.write(
                f"**{score_type}**: {submission_names[max_idx]} = {max(score_values):.2%}"
            )


def main():
    st.set_page_config(page_title="Copyadi Checker", page_icon="🔍", layout="wide")

    st.title("🔍 Copyadi Checker")
    st.markdown("Compare new submissions against batch uploads of previous submissions")



    # ===== Sidebar Logic =====

    # 1. Check API Connection (Top)
    api_connected = False
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=1)
        if response.status_code == 200:
            api_connected = True
        else:
            st.sidebar.error(f"❌ API Error: {response.status_code}")
    except:
        st.sidebar.error("❌ API Not Running")
        st.sidebar.info(f"Start API: `{API_BASE_URL}`")

    
    if api_connected:
        if not is_logged_in():
            # --- Not Logged In: Show Login First ---
            st.sidebar.header("🔐 Account")
            auth_tab1, auth_tab2 = st.sidebar.tabs(["🔑 Login", "📝 Register"])
            
            with auth_tab1:
                with st.form("login_form"):
                    username = st.text_input("Username", key="login_username")
                    password = st.text_input("Password", type="password", key="login_password")
                    submit = st.form_submit_button("Login", type="primary")

                    if submit and username and password:
                        if login_user(username, password):
                            st.rerun()
            
            with auth_tab2:
                with st.form("register_form"):
                    reg_username = st.text_input("Username", key="reg_username")
                    reg_password = st.text_input("Password", type="password", key="reg_password")
                    reg_confirm = st.text_input("Confirm Password", type="password", key="reg_confirm")
                    invite_code = st.text_input("Invite Code", key="invite_code", 
                                            help="Required. Get this from an admin.")
                    reg_submit = st.form_submit_button("Register")
                    
                    if reg_submit:
                        if not reg_username or not reg_password:
                            st.warning("Please fill all fields")
                        elif reg_password != reg_confirm:
                            st.error("Passwords don't match")
                        elif not invite_code:
                            st.error("Invite code is required")
                        else:
                            if register_user(reg_username, reg_password, invite_code):
                                pass  # Success message shown in function
            
            st.sidebar.divider()

        else:
            # --- Logged In: Show Workspaces First ---
            st.sidebar.subheader("📦 Workspaces")
            
            instances = fetch_instances()
            
            if instances:
                # Create display names for selector
                instance_options = {inst["instance_id"]: f"{inst['name']}" for inst in instances}
                
                # Initialize current instance if not set
                if "current_instance_id" not in st.session_state:
                    st.session_state.current_instance_id = instances[0]["instance_id"]
                
                # Instance selector
                selected_id = st.sidebar.selectbox(
                    "Select Workspace",
                    options=list(instance_options.keys()),
                    format_func=lambda x: instance_options.get(x, x),
                    index=list(instance_options.keys()).index(st.session_state.current_instance_id) 
                        if st.session_state.current_instance_id in instance_options else 0,
                    key="instance_selector"
                )
                
                # If instance changed, update state
                if selected_id != st.session_state.current_instance_id:
                    st.session_state.current_instance_id = selected_id
                    st.rerun()
                
                # Show current instance info
                current_inst = next((i for i in instances if i["instance_id"] == selected_id), None)
                if current_inst:
                    if current_inst.get("description"):
                        st.sidebar.caption(current_inst["description"])
                    
                    # Show stored URLs for this instance
                    stored_urls = current_inst.get("github_urls", [])
                    if stored_urls:
                        with st.sidebar.expander(f"📂 Stored URLs ({len(stored_urls)})", expanded=False):
                            for url in stored_urls:
                                st.markdown(f"- [{url.split('/')[-1]}]({url})")
            else:
                st.sidebar.info("No workspaces found")
                st.session_state.current_instance_id = None
            
            # Create new instance
            with st.sidebar.expander("➕ New Workspace", expanded=False):
                new_name = st.text_input("Name", key="new_inst_name")
                new_desc = st.text_input("Description", key="new_inst_desc")
                if st.button("Create", key="create_inst_btn"):
                    if new_name:
                        result = create_new_instance(new_name, new_desc)
                        if result:
                            st.success(f"Created: {new_name}")
                            st.session_state.current_instance_id = result["instance_id"]
                            st.rerun()
                    else:
                        st.warning("Enter a name")
            
            # Delete current instance
            if instances and len(instances) > 1:
                if st.sidebar.button("🗑️ Delete Current Workspace", type="secondary"):
                    if delete_instance_api(st.session_state.current_instance_id):
                        st.session_state.current_instance_id = None
                        st.rerun()
            
            st.sidebar.divider()

        # --- Configuration (Central) ---
        st.sidebar.header("Configuration")

        # Preprocessing options
        st.sidebar.subheader("Preprocessing Options")
        remove_comments = st.sidebar.checkbox("Remove Comments", value=True)
        normalize_whitespace = st.sidebar.checkbox("Normalize Whitespace", value=True)
        preserve_variable_names = st.sidebar.checkbox("Preserve Variable Names", value=True)
        preserve_literals = st.sidebar.checkbox("Preserve Literals", value=False)

        # OS junk patterns - always filtered
        OS_JUNK_PATTERNS = [
            ".DS_Store",
            "._*",
            "__MACOSX",
            "Thumbs.db",
            "desktop.ini",
            "$RECYCLE.BIN",
            ".Trash-*",
        ]

        user_ignore_patterns = st.sidebar.multiselect(
            "Ignore Patterns",
            ["*.xlsx", "*.pdf", "*.git", "__pycache__", "node_modules", "*.pyc", "*.log"],
            default=["*.xlsx", "*.pdf", "*.git", "__pycache__"],
            accept_new_options=True,
        )

        ignore_patterns = OS_JUNK_PATTERNS + user_ignore_patterns

        preprocessing_options = {
            "remove_comments": remove_comments,
            "normalize_whitespace": normalize_whitespace,
            "preserve_variable_names": preserve_variable_names,
            "preserve_literals": preserve_literals,
        }

        st.sidebar.divider()

        # --- Footer: User Info & API Status ---
        
        if is_logged_in():
            st.sidebar.header("🔐 Account")
            st.sidebar.success(f"Logged in as: **{st.session_state.username}**")
            if st.sidebar.button("Logout"):
                logout_user()
                st.rerun()
            st.sidebar.divider()

        # Success message at bottom
        st.sidebar.success(f"✅ API Connected to {API_BASE_URL}")
    
    # Initialize session state
    if "all_submissions" not in st.session_state:
        st.session_state.all_submissions = {}
    if "preprocessed_embeddings" not in st.session_state:
        st.session_state.preprocessed_embeddings = None
    if "is_preprocessed" not in st.session_state:
        st.session_state.is_preprocessed = False

    # Main content - Three input methods
    st.header("📥 Input Methods")

    tab1, tab2, tab3 = st.tabs(
        ["🔗 URL Analysis (GitHub, PDF, etc.)", "📦 ZIP File Upload", "📄 Individual Files"]
    )

    with tab1:
        # Get stored URLs from current instance
        current_instance_id = st.session_state.get("current_instance_id")
        stored_urls = []
        if current_instance_id:
            instances = fetch_instances()
            current_inst = next((i for i in instances if i["instance_id"] == current_instance_id), None)
            if current_inst:
                stored_urls = current_inst.get("github_urls", [])
        
        # Pre-fill text area with stored URLs
        default_urls = "\n".join(stored_urls) if stored_urls else ""
        
        github_urls_text = st.text_area(
            "Enter URLs (GitHub repos, PDF links, direct files):",
            value=default_urls,
            placeholder="https://github.com/user/repo\nhttps://example.com/paper.pdf",
            height=150,
            key="github_urls_input"
        )

        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Run Analysis", key="direct_analysis_github", type="primary", use_container_width=True):
                if github_urls_text:
                    urls = [url.strip() for url in github_urls_text.split("\n") if url.strip()]
                    
                    # Auto-save URLs
                    if current_instance_id:
                        update_instance_urls_api(current_instance_id, urls)
                    
                    # Process URLs
                    with st.spinner(f"Processing {len(urls)} URLs..."):
                        st.session_state.all_submissions = process_github_via_api(
                            urls, ignore_patterns
                        )
                    
                    # Run analysis immediately
                    if st.session_state.all_submissions:
                        with st.spinner("Analyzing all submissions..."):
                            target = next(iter(st.session_state.all_submissions.keys()))
                            result = analyze_direct_via_api(
                                st.session_state.all_submissions, target, preprocessing_options
                            )
                        if result:
                            st.session_state.analysis_result = result
                    else:
                        st.error("No submissions extracted")
                else:
                    st.warning("Please enter at least one URL")
        
        with col2:
            if st.button("� Process Only", key="process_github", use_container_width=True):
                if github_urls_text:
                    urls = [url.strip() for url in github_urls_text.split("\n") if url.strip()]
                    
                    # Auto-save URLs to current workspace
                    if current_instance_id:
                        update_instance_urls_api(current_instance_id, urls)
                    
                    with st.spinner(f"Processing {len(urls)} URLs..."):
                        st.session_state.all_submissions = process_github_via_api(
                            urls, ignore_patterns
                        )
                        st.session_state.is_preprocessed = False
                        st.session_state.preprocessed_embeddings = None
                    st.success(f"✅ Loaded {len(st.session_state.all_submissions)} submissions")
                else:
                    st.warning("Please enter at least one URL")

    with tab2:
        zip_file = st.file_uploader(
            "Upload ZIP file containing all previous submissions",
            type=["zip"],
            key="zip_upload",
        )

        if zip_file:
            with st.spinner("Uploading to API and extracting..."):
                st.session_state.all_submissions = upload_files_to_api(
                    [zip_file], ignore_patterns
                )
                st.session_state.is_preprocessed = False
                st.session_state.preprocessed_embeddings = None

            st.success(
                f"✅ Loaded {len(st.session_state.all_submissions)} submissions from ZIP"
            )

    with tab3:
        st.info("Upload individual submissions - each file/ZIP = one submission")

        prev_submissions = st.file_uploader(
            "Upload submissions one by one",
            type=[
                "py",
                "zip",
                "java",
                "cpp",
                "c",
                "js",
                "txt",
                "pdf",
                "docx",
                "doc",
                "html",
                "md",
                "json",
                "xml",
                "csv",
                "rtf",
            ],
            accept_multiple_files=True,
            key="individual_files",
        )

        if prev_submissions:
            with st.spinner("Uploading to API..."):
                st.session_state.all_submissions = upload_individual_to_api(
                    prev_submissions, ignore_patterns
                )
                st.session_state.is_preprocessed = False
                st.session_state.preprocessed_embeddings = None
            st.success(f"✅ Loaded {len(st.session_state.all_submissions)} submissions")

    all_submissions = st.session_state.all_submissions

    # ===== Target Selection (shows after Process Only) =====
    if all_submissions:
        st.divider()
        st.subheader("🎯 Select Target & Analyze")
        
        file_names = list(all_submissions.keys())
        target_file = st.selectbox(
            "Select target file to compare against others:",
            file_names,
            key="target_file"
        )

        if st.button("🔍 Analyze Selected", use_container_width=True):
            with st.spinner("Analyzing all submissions..."):
                result = analyze_direct_via_api(
                    all_submissions, target_file, preprocessing_options
                )
            if result:
                st.session_state.analysis_result = result

    # ===== Results Display =====
    if "analysis_result" in st.session_state and st.session_state.analysis_result:
        st.divider()
        display_results(st.session_state.analysis_result)


if __name__ == "__main__":
    main()
