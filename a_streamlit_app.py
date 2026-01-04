"""
Streamlit Frontend for Code Similarity Checker
Communicates with api_server.py via HTTP requests.

Two modes:
- Direct: Upload → Select target → Analyze (one-pass, memory efficient)
- Preprocess: Upload → Preprocess → Select target → Compare (keeps embeddings)
"""

import requests
import streamlit as st
from collections import Counter

# API endpoint - tries localhost first, falls back to remote
try:
    requests.get("http://localhost:7860/", timeout=1)
    API_BASE_URL = "http://localhost:7860"
except:
    API_BASE_URL = "https://arnavam-copyadi-finder.hf.space"


# ===== Auth Helper Functions =====


def calculate_overall_match(score_values):
    """
    Calculate overall score using Mode/Max logic:
    - If max frequency == 1 (all unique): take max score
    - If mode exists: take highest mode
    """
    if not score_values:
        return 0
        
    counts = Counter(score_values)
    max_freq = max(counts.values())
    
    if max_freq == 1:
        # All unique -> take highest score
        return max(score_values)
    else:
        # Mode exists -> take highest mode (in case of tie)
        modes = [k for k, v in counts.items() if v == max_freq]
        return max(modes)


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
            st.success("✅ Registered! Please login with your credentials.")
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
            f"{API_BASE_URL}/extract/upload",
            files=file_data,
            params=params,
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
            f"{API_BASE_URL}/extract/urls",
            json={"urls": urls, "ignore_patterns": ignore_patterns},
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


def fetch_instance_details(instance_id):
    """Fetch single instance details - this triggers Discord URL scraping."""
    if not is_logged_in():
        return None

    try:
        response = requests.get(
            f"{API_BASE_URL}/instances/{instance_id}",
            headers=get_auth_headers(),
            timeout=30,  # Discord scraping may take time
        )
        response.raise_for_status()
        return response.json().get("instance")
    except Exception as e:
        print(f"Error fetching instance details: {e}")
        return None


def fetch_instances():
    """Fetch user's instances from API (basic list, no Discord scraping)."""
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


def create_new_instance(name, description="", discord_channel_id=None):
    """Create a new instance via API. Optionally link to Discord channel."""
    if not is_logged_in():
        return None

    try:
        params = {"name": name, "description": description}
        if discord_channel_id:
            params["discord_channel_id"] = discord_channel_id
        
        response = requests.post(
            f"{API_BASE_URL}/instances",
            params=params,
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
                "submissions": submissions if not st.session_state.get("buffer_id") else None,
                "buffer_id": st.session_state.get("buffer_id"),
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


def analyze_hybrid_via_api(submissions, target_file, preprocessing_options, instance_id):
    """Hybrid analysis: Vector Search -> Prune -> Direct (uses LLM embeddings)."""
    if not is_logged_in():
        st.error("Please login first")
        return None

    try:
        response = requests.post(
            f"{API_BASE_URL}/analyze",
            json={
                "submissions": submissions if not st.session_state.get("buffer_id") else None,
                "buffer_id": st.session_state.get("buffer_id"),
                "target_file": target_file,
                "preprocessing_options": preprocessing_options,
                "mode": "hybrid",
                "instance_id": instance_id,
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
                "submissions": submissions if not st.session_state.get("buffer_id") else None,
                "buffer_id": st.session_state.get("buffer_id"),
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
        return response.json()
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


def get_similar_regions_via_api(file1_name, file1_content, file2_name, file2_content):
    """Get detailed similar regions between two files."""
    if not is_logged_in():
        st.error("Please login first")
        return None

    try:
        response = requests.post(
            f"{API_BASE_URL}/similar-regions",
            json={
                "buffer_id": st.session_state.get("buffer_id"),
                "file1_name": file1_name,
                "file1_content": file1_content,
                "file2_name": file2_name,
                "file2_content": file2_content,
                "block_threshold": 0.6,
                "min_tokens": 10,
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


def display_results(result):
    """Display similarity analysis results (Table Only)."""
    scores = result.get("scores", {})
    submission_names = result.get("submission_names", [])

    if not scores or not submission_names:
        st.warning("No results to display")
        return

    # 1. Calculate Aggregates
    data = []
    for i, name in enumerate(submission_names):
        row = {"Submission": name}
        total_score = 0
        count = 0

        for cat, values in scores.items():
            if values and i < len(values):
                # Normalize key name (e.g., "raw_scores" -> "Raw")
                clean_cat = cat.replace("_scores", "").title()
                val = values[i]
                row[clean_cat] = val
                total_score += val
                count += 1
                
                # Collect values for mode/max calculation
                if "score_values" not in row:
                    row["score_values"] = []
                row["score_values"].append(val)

        # Calculate overall score using shared helper
        row["Overall Match"] = calculate_overall_match(row.get("score_values", []))
        
        # Cleanup temp list
        if "score_values" in row:
            del row["score_values"]

        data.append(row)

    # Sort by Overall Match
    data.sort(key=lambda x: x["Overall Match"], reverse=True)

    # 2. Display Table
    st.header("📊 Similarity Metrics")

    if len(data) > 0:
        st.dataframe(
            data,
            use_container_width=True,
            column_config={
                "Overall Match": st.column_config.ProgressColumn(
                    "Overall Match",
                    format="%.1f%%",
                    min_value=0,
                    max_value=1,
                ),
                "Submission": st.column_config.TextColumn("File Name"),
                "Raw": st.column_config.NumberColumn("Raw Text", format="%.1f%%"),
                "Processed": st.column_config.NumberColumn(
                    "Processed", format="%.1f%%"
                ),
                "Ast": st.column_config.NumberColumn(
                    "Structure (AST)", format="%.1f%%"
                ),
                "Token": st.column_config.NumberColumn("Token Seq", format="%.1f%%"),
                "Cosine": st.column_config.NumberColumn(
                    "Cosine (TF-IDF)", format="%.1f%%"
                ),
            },
        )


def display_similar_regions(regions_result, file1_name, file2_name):
    """Display detailed similar regions with polished side-by-side code blocks."""
    st.markdown("---")
    st.header(f"⚖️ Comparison: {file1_name} vs {file2_name}")

    # Score Dashboard
    stats = regions_result.get("stats", {})
    overall_sim = regions_result.get("overall_similarity", 0)

    # Color logic
    sim_color = (
        "red" if overall_sim > 0.7 else "orange" if overall_sim > 0.4 else "green"
    )

    # Top stats row
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Overall Similarity", f"{overall_sim:.1%}")
    kpi2.metric("Function Matches", stats.get("total_function_matches", 0))
    kpi3.metric("Token Matches", stats.get("total_token_matches", 0))
    kpi4.metric("Process Time", f"{stats.get('analysis_time', 0):.3f}s")

    st.markdown("---")

    # 1. Structural Matches (Functions/Classes)
    function_matches = regions_result.get("function_matches", [])

    with st.expander(
        f"🧩 Similar Functions & Classes ({len(function_matches)})", expanded=True
    ):
        if function_matches:
            st.caption(
                "Matches found via AST analysis (resilient to renamed variables)."
            )

            for match in function_matches[:10]:
                similarity = match["similarity"]

                # Visual separator
                st.markdown(
                    f"##### {match['type'].title()}: `{match['file1_block']}` ↔ `{
                        match['file2_block']
                    }`"
                )

                # Progress bar for this specific match
                c_score, c_bar = st.columns([1, 5])
                with c_score:
                    st.markdown(f"**{similarity:.1%}** match")
                with c_bar:
                    bar_color = (
                        ":red["
                        if similarity > 0.8
                        else ":orange["
                        if similarity > 0.6
                        else ":green["
                    )
                    st.progress(similarity)

                # Comparison View
                c_left, c_right = st.columns(2)
                with c_left:
                    st.markdown(
                        f"**{file1_name}** (L{match['file1_lines'][0]}-{
                            match['file1_lines'][1]
                        })"
                    )
                    st.code(match["file1_source"], language="python")

                with c_right:
                    st.markdown(
                        f"**{file2_name}** (L{match['file2_lines'][0]}-{
                            match['file2_lines'][1]
                        })"
                    )
                    st.code(match["file2_source"], language="python")

                st.divider()
        else:
            st.info("No structural matches found.")

    # 2. Exact/Token Matches
    token_matches = regions_result.get("token_matches", [])

    with st.expander(f"📝 Exact Token Sequences ({len(token_matches)})", expanded=True):
        if token_matches:
            st.caption(
                "Matches found via token sequence analysis (copy-paste detection)."
            )

            # Show top matches in a cleaner grid
            for i, match in enumerate(token_matches[:10]):
                st.markdown(
                    f"**Match #{i + 1}**: {match['token_count']} matching tokens"
                )

                c1, c2 = st.columns(2)
                c1.info(f"📍 {file1_name}: Line ~{match['file1_approx_line']}")
                c2.info(f"📍 {file2_name}: Line ~{match['file2_approx_line']}")

                st.code(match["matched_text"], language="python")
                st.markdown("<br>", unsafe_allow_html=True)
        else:
            st.info("No token sequences found.")


def main():
    st.set_page_config(page_title="Copyadi Checker", page_icon="🔍", layout="wide")

    st.title("🔍 Copyadi Checker")
    st.markdown("Compare new submissions against batch uploads of previous submissions")

    ignore_patterns = None

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
                    password = st.text_input(
                        "Password", type="password", key="login_password"
                    )
                    submit = st.form_submit_button("Login", type="primary")

                    if submit and username and password:
                        if login_user(username, password):
                            st.rerun()

            with auth_tab2:
                with st.form("register_form"):
                    reg_username = st.text_input("Username", key="reg_username")
                    reg_password = st.text_input(
                        "Password", type="password", key="reg_password"
                    )
                    reg_confirm = st.text_input(
                        "Confirm Password", type="password", key="reg_confirm"
                    )
                    invite_code = st.text_input(
                        "Invite Code",
                        key="invite_code",
                        help="Required. Get this from an admin.",
                    )
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
                instance_options = {
                    inst["instance_id"]: f"{inst['name']}" for inst in instances
                }

                # Initialize current instance if not set
                if "current_instance_id" not in st.session_state:
                    st.session_state.current_instance_id = instances[0]["instance_id"]

                # Instance selector
                selected_id = st.sidebar.selectbox(
                    "Select Workspace",
                    options=list(instance_options.keys()),
                    format_func=lambda x: instance_options.get(x, x),
                    index=list(instance_options.keys()).index(
                        st.session_state.current_instance_id
                    )
                    if st.session_state.current_instance_id in instance_options
                    else 0,
                    key="instance_selector",
                )

                # If instance changed, update state
                if selected_id != st.session_state.current_instance_id:
                    st.session_state.current_instance_id = selected_id
                    st.rerun()

                # Show current instance info
                current_inst = next(
                    (i for i in instances if i["instance_id"] == selected_id), None
                )
                if current_inst:
                    if current_inst.get("description"):
                        st.sidebar.caption(current_inst["description"])

                    # Show stored URLs for this instance
                    stored_urls = current_inst.get("github_urls", [])
                    if stored_urls:
                        with st.sidebar.expander(
                            f"📂 Stored URLs ({len(stored_urls)})", expanded=False
                        ):
                            for url in stored_urls:
                                st.markdown(f"[{url.split('/')[-1]}]({url})")
                    
                    # Add Discord sync button if workspace has Discord channel
                    if current_inst.get("discord_channel_id"):
                        if st.sidebar.button("🔄 Sync Discord URLs", key="sync_discord_btn"):
                            with st.spinner("Fetching URLs from Discord..."):
                                details = fetch_instance_details(current_inst["instance_id"])
                                if details and details.get("github_urls"):
                                    # Update session state so URLs appear in text area
                                    st.session_state.github_urls_input = "\n".join(details["github_urls"])
                                    st.sidebar.success(f"✅ Loaded {len(details['github_urls'])} URLs!")
                                    st.rerun()
                                else:
                                    st.sidebar.warning("No URLs found with matching tag")
            else:
                st.sidebar.info("No workspaces found")
                st.session_state.current_instance_id = None

            # Create new instance
            with st.sidebar.expander("➕ New Workspace", expanded=False):
                new_name = st.text_input("Name", key="new_inst_name",
                    help="Use Discord #tag name (e.g., 'nlp-classification')")
                new_desc = st.text_input("Description", key="new_inst_desc")
                discord_channel = st.text_input(
                    "Discord Channel ID (optional)", 
                    key="discord_channel_id",
                    help="If set, URLs will be auto-loaded from Discord #tag"
                )
                if st.button("Create", key="create_inst_btn"):
                    if new_name:
                        result = create_new_instance(
                            new_name, 
                            new_desc, 
                            discord_channel if discord_channel else None
                        )
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
        preserve_variable_names = st.sidebar.checkbox(
            "Preserve Variable Names", value=True
        )
        anonymize_literals = st.sidebar.checkbox("Anonymize Literals", value=True)
        
        # LLM Mode Toggle
        st.sidebar.subheader("Analysis Mode")
        use_llm = st.sidebar.toggle(
            "🧠 Use LLM Embeddings", 
            value=False,
            help="Uses AI embeddings for faster search on large datasets. Requires Pinecone API key."
        )

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
            [
                "*.xlsx",
                "*.pdf",
                "*.git",
                "__pycache__",
                "node_modules",
                "*.pyc",
                "*.log",
            ],
            default=["*.xlsx", "*.pdf", "*.git", "__pycache__"],
            accept_new_options=True,
        )

        ignore_patterns = OS_JUNK_PATTERNS + user_ignore_patterns

        preprocessing_options = {
            "remove_comments": remove_comments,
            "normalize_whitespace": normalize_whitespace,
            "preserve_variable_names": preserve_variable_names,
            "preserve_literals": not anonymize_literals,
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
    if "buffer_id" not in st.session_state:
        st.session_state.buffer_id = None
    if "filenames" not in st.session_state:
        st.session_state.filenames = []
    if "preprocessed_embeddings" not in st.session_state:
        st.session_state.preprocessed_embeddings = None
    if "is_preprocessed" not in st.session_state:
        st.session_state.is_preprocessed = False

    # Main content - Three input methods
    st.header("📥 Input Methods")

    tab1, tab2, tab3 = st.tabs(
        [
            "🔗 URL Analysis (GitHub, PDF, etc.)",
            "📦 ZIP File Upload",
            "📄 Individual Files",
        ]
    )

    with tab1:
        # Get stored URLs from current instance
        current_instance_id = st.session_state.get("current_instance_id")
        stored_urls = []
        if current_instance_id:
            instances = fetch_instances()
            current_inst = next(
                (i for i in instances if i["instance_id"] == current_instance_id), None
            )
            if current_inst:
                stored_urls = current_inst.get("github_urls", [])

        # Initialize session state for URL input if not set
        if "github_urls_input" not in st.session_state:
            st.session_state.github_urls_input = "\n".join(stored_urls) if stored_urls else ""

        # Add Load from Workspace button
        if st.button("📥 Load from Workspace", key="load_workspace_urls", use_container_width=False):
            if current_instance_id and stored_urls:
                st.session_state.github_urls_input = "\n".join(stored_urls)
                st.rerun()
            elif not current_instance_id:
                st.warning("No workspace selected")
            else:
                st.info("No URLs stored in current workspace")

        github_urls_text = st.text_area(
            "Enter URLs (GitHub repos, PDF links, direct files):",
            placeholder="https://github.com/user/repo\nhttps://example.com/paper.pdf",
            height=150,
            key="github_urls_input",
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "🚀 Run Analysis",
                key="direct_analysis_github",
                type="primary",
                use_container_width=True,
            ):
                if github_urls_text:
                    urls = [
                        url.strip()
                        for url in github_urls_text.split("\n")
                        if url.strip()
                    ]

                    # Auto-save URLs only if changed
                    if current_instance_id and sorted(urls) != sorted(stored_urls):
                        update_instance_urls_api(current_instance_id, urls)

                    # Process URLs
                    with st.spinner(f"Processing {len(urls)} URLs..."):
                        result_data = process_github_via_api(urls, ignore_patterns)
                        if result_data:
                            st.session_state.buffer_id = result_data["buffer_id"]
                            st.session_state.filenames = result_data["filenames"]
                            st.session_state.all_submissions = {name: "" for name in result_data["filenames"]}
                        else:
                            st.session_state.all_submissions = {}


                    # Run analysis immediately
                    if st.session_state.all_submissions:
                        with st.spinner("Analyzing all submissions..."):
                            target = next(iter(st.session_state.all_submissions.keys()))
                            if use_llm and st.session_state.get("current_instance_id"):
                                result = analyze_hybrid_via_api(
                                    st.session_state.all_submissions,
                                    target,
                                    preprocessing_options,
                                    st.session_state.current_instance_id,
                                )
                            else:
                                result = analyze_direct_via_api(
                                    st.session_state.all_submissions,
                                    target,
                                    preprocessing_options,
                                )
                        if result:
                            st.session_state.analysis_result = result
                    else:
                        st.error("No submissions extracted")
                else:
                    st.warning("Please enter at least one URL")

        with col2:
            if st.button(
                "� Process Only", key="process_github", use_container_width=True
            ):
                if github_urls_text:
                    urls = [
                        url.strip()
                        for url in github_urls_text.split("\n")
                        if url.strip()
                    ]

                    # Auto-save URLs to current workspace only if changed
                    if current_instance_id and sorted(urls) != sorted(stored_urls):
                        update_instance_urls_api(current_instance_id, urls)

                    with st.spinner(f"Processing {len(urls)} URLs..."):
                        result_data = process_github_via_api(urls, ignore_patterns)
                        if result_data:
                            st.session_state.buffer_id = result_data["buffer_id"]
                            st.session_state.filenames = result_data["filenames"]
                            st.session_state.all_submissions = {name: "" for name in result_data["filenames"]}
                        else:
                            st.session_state.all_submissions = {}

                        st.session_state.is_preprocessed = False
                        st.session_state.preprocessed_embeddings = None
                    st.success(
                        f"✅ Loaded {len(st.session_state.all_submissions)} submissions"
                    )
                else:
                    st.warning("Please enter at least one URL")

    with tab2:
        zip_file = st.file_uploader(
            "Upload ZIP file containing all previous submissions",
            type=["zip"],
            key="zip_upload",
        )

        if zip_file:
            with st.spinner("Extracting text from ZIP..."):
                result_data = upload_files_to_api([zip_file], ignore_patterns)
                if result_data:
                    st.session_state.buffer_id = result_data["buffer_id"]
                    st.session_state.filenames = result_data["filenames"]
                    st.session_state.all_submissions = {name: "" for name in result_data["filenames"]}
                else:
                    st.session_state.all_submissions = {}
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
            with st.spinner(f"Processing {len(prev_submissions)} files..."):
                result_data = upload_individual_to_api(prev_submissions, ignore_patterns)
                if result_data:
                    st.session_state.buffer_id = result_data["buffer_id"]
                    st.session_state.filenames = result_data["filenames"]
                    st.session_state.all_submissions = {name: "" for name in result_data["filenames"]}
                else:
                    st.session_state.all_submissions = {}
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
            key="target_file",
        )

        if st.button("🔍 Analyze Selected", use_container_width=True):
            with st.spinner("Analyzing all submissions..."):
                if use_llm and st.session_state.get("current_instance_id"):
                    result = analyze_hybrid_via_api(
                        all_submissions, target_file, preprocessing_options,
                        st.session_state.current_instance_id
                    )
                else:
                    result = analyze_direct_via_api(
                        all_submissions, target_file, preprocessing_options
                    )
            if result:
                st.session_state.analysis_result = result

    # ===== Results Display =====
    if "analysis_result" in st.session_state and st.session_state.analysis_result:
        st.divider()
        display_results(st.session_state.analysis_result)

        # ===== Similar Regions Detail View =====
        result = st.session_state.analysis_result
        if result.get("submission_names") and all_submissions:
            st.divider()
            st.subheader("🔬 Detailed Comparison")
            st.caption("View which specific functions/code blocks are similar")

            # Find the most similar file
            scores = result.get("scores", {})
            submission_names = result.get("submission_names", [])
            target_file = result.get("target_file", "")

            # Calculate overall score for each submission using Mode/Max logic
            final_scores = []
            for i, name in enumerate(submission_names):
                score_values = []
                for score_type, values in scores.items():
                    if values and i < len(values):
                        score_values.append(values[i])
                
                final_scores.append(calculate_overall_match(score_values))

            # Sort by score (descending)
            sorted_submissions = sorted(
                zip(submission_names, final_scores), key=lambda x: -x[1]
            )

            # Let user select which file to compare in detail
            compare_options = [
                f"{name} ({score:.1%})" for name, score in sorted_submissions
            ]
            selected_compare = st.selectbox(
                "Select file to compare with target:",
                compare_options,
                key="compare_detail_select",
            )

            # Extract selected file name
            selected_file = sorted_submissions[compare_options.index(selected_compare)][
                0
            ]

            if st.button(
                "🔬 View Similar Regions", use_container_width=True, type="primary"
            ):
                if target_file in all_submissions and selected_file in all_submissions:
                    with st.spinner("Analyzing code regions..."):
                        regions_result = get_similar_regions_via_api(
                            target_file,
                            None, # No content needed if buffer_id is available
                            selected_file,
                            None,
                        )
                    if regions_result:
                        st.session_state.regions_result = regions_result
                        st.session_state.regions_files = (target_file, selected_file)
                else:
                    st.error("Cannot find file content for comparison")

        # Display similar regions if available
        if "regions_result" in st.session_state and st.session_state.regions_result:
            file1, file2 = st.session_state.get("regions_files", ("File 1", "File 2"))
            display_similar_regions(st.session_state.regions_result, file1, file2)



if __name__ == "__main__":
    main()
