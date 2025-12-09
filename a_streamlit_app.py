"""
Streamlit Frontend for Code Similarity Checker
Communicates with api_server.py via HTTP requests.
"""

import numpy as np
import requests
import streamlit as st

# API endpoint - change this if running API on different host/port
API_BASE_URL = "http://localhost:8000"


def upload_files_to_api(files, ignore_patterns=None):
    """Upload files to API and get extracted text back."""
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
            params=params
        )
        response.raise_for_status()
        return response.json()["submissions"]
    except Exception as e:
        st.error(f"API Error: {e}")
        return {}


def process_github_via_api(urls, ignore_patterns=None):
    """Process GitHub URLs via API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/github",
            json={
                "urls": urls,
                "ignore_patterns": ignore_patterns
            }
        )
        response.raise_for_status()
        return response.json()["submissions"]
    except Exception as e:
        st.error(f"API Error: {e}")
        return {}


def analyze_via_api(submissions, target_file, preprocessing_options):
    """Calculate similarity via API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/analyze",
            json={
                "submissions": submissions,
                "target_file": target_file,
                "preprocessing_options": preprocessing_options
            }
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


def upload_individual_to_api(files, ignore_patterns=None):
    """Upload files where each file/ZIP = ONE submission."""
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
            params=params
        )
        response.raise_for_status()
        return response.json()["submissions"]
    except Exception as e:
        st.error(f"API Error: {e}")
        return {}


def main():
    st.set_page_config(
        page_title="Copyadi Checker", page_icon="🔍", layout="wide"
    )
    
    st.title("🔍 Copyadi Checker")
    st.markdown("Compare new submissions against batch uploads of previous submissions")

    # API connection status
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=2)
        if response.status_code == 200:
            st.sidebar.success("✅ API Connected")
        else:
            st.sidebar.error("❌ API Error")
    except:
        st.sidebar.error("❌ API Not Running")
        st.sidebar.info(f"Start API: `uvicorn api_server:app --reload`")

    # Sidebar for configuration
    st.sidebar.header("Configuration")

    # Preprocessing options
    st.sidebar.subheader("Preprocessing Options")
    remove_comments = st.sidebar.checkbox("Remove Comments", value=True)
    normalize_whitespace = st.sidebar.checkbox("Normalize Whitespace", value=True)
    preserve_variable_names = st.sidebar.checkbox("Preserve Variable Names", value=True)
    preserve_literals = st.sidebar.checkbox("Preserve Literals", value=False)

    ignore_patterns = st.sidebar.multiselect(
        "Ignore Patterns",
        ["*.xlsx", "*.pdf", "*.git", "__pycache__", ".DS_Store", "node_modules"],
        default=["*.xlsx", "*.pdf", "*.git", "__pycache__", ".DS_Store"],
        accept_new_options=True,
    )

    preprocessing_options = {
        "remove_comments": remove_comments,
        "normalize_whitespace": normalize_whitespace,
        "preserve_variable_names": preserve_variable_names,
        "preserve_literals": preserve_literals,
    }

    # Main content - Three input methods
    st.header("📥 Input Methods")

    tab1, tab2, tab3 = st.tabs(
        ["📦 ZIP File Upload", "🔗 Multiple GitHub URLs", "📄 Individual Files"]
    )

    # Store submissions in session state
    if "all_submissions" not in st.session_state:
        st.session_state.all_submissions = {}

    with tab1:
        zip_file = st.file_uploader(
            "Upload ZIP file containing all previous submissions",
            type=["zip"],
            key="zip_upload",
        )

        if zip_file:
            with st.spinner("Uploading to API and extracting..."):
                st.session_state.all_submissions = upload_files_to_api([zip_file], ignore_patterns)

            st.success(
                f"✅ Processed {len(st.session_state.all_submissions)} submissions from ZIP"
            )

    with tab2:
        github_urls_text = st.text_area(
            "Enter GitHub URLs (one per line):",
            placeholder="https://github.com/user/repo1\nhttps://github.com/user/repo2",
            height=150,
        )

        if st.button("Process GitHub URLs", key="process_github"):
            if github_urls_text:
                urls = [
                    url.strip() for url in github_urls_text.split("\n") if url.strip()
                ]
                with st.spinner(f"Processing {len(urls)} GitHub repositories..."):
                    st.session_state.all_submissions = process_github_via_api(urls, ignore_patterns)
                st.success(f"✅ Processed {len(st.session_state.all_submissions)} submissions")
            else:
                st.warning("Please enter at least one GitHub URL")

    with tab3:
        st.info("Upload individual submissions - each file/ZIP = one submission")

        prev_submissions = st.file_uploader(
            "Upload submissions one by one",
            type=["py", "zip", "java", "cpp", "c", "js", "txt", "pdf", "docx", "doc", "html", "md", "json", "xml", "csv", "rtf"],
            accept_multiple_files=True,
            key="individual_files",
        )

        if prev_submissions:
            with st.spinner("Uploading to API..."):
                st.session_state.all_submissions = upload_individual_to_api(prev_submissions, ignore_patterns)
            st.success(f"✅ Loaded {len(st.session_state.all_submissions)} submissions")

    # New submission input
    st.header("🎯 New Submission to Check")
    
    all_submissions = st.session_state.all_submissions
    
    if all_submissions:
        file_names = list(all_submissions.keys())
        target_file = st.selectbox("Select file to check:", file_names)
    else:
        target_file = None
        st.info("Upload submissions first")

    # Analysis section
    if st.button("🚀 Analyze Similarity", type="primary", use_container_width=True):
        if not all_submissions:
            st.error("❌ Please upload submissions first")
            return

        if not target_file:
            st.error("❌ Please select a file to check")
            return

        with st.spinner("Analyzing via API..."):
            result = analyze_via_api(all_submissions, target_file, preprocessing_options)

        if result:
            # Display results
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
                    max_score = max([max(v) for v in scores.values() if v])
                    st.metric("Highest Similarity", f"{max_score:.2%}")

            # Results table
            st.subheader("Similar Submissions Found")
            for score_type, score_values in scores.items():
                if score_values:
                    max_idx = np.argmax(score_values)
                    st.write(f"**{score_type}**: {submission_names[max_idx]} = {max(score_values):.2%}")


if __name__ == "__main__":
    main()
