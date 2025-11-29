import os
import tempfile
import zipfile
from pathlib import Path
from typing import List

import chardet
import git
import numpy as np
import streamlit as st

# Import your existing classes
from code_similarity_finder import (
    FlexibleCodeSimilarityChecker,
    process_all_submissions,
)

def detect_encoding(file_obj):
    """Detect encoding from UploadedFile bytes"""
    raw_data = file_obj.getvalue()
    file_obj.seek(0)
    return chardet.detect(raw_data)["encoding"]


def extract_zip(uploaded_zip, extract_to):
    """Extract uploaded zip file to temporary directory"""
    with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    return extract_to


def process_github_links(github_urls: List[str], ignore_patterns=None):
    all_submissions = {}

    for i, url in enumerate(github_urls):
        if url.strip():
            try:
                repo_name = url.split("/")[-1].replace(".git", "")
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Clone repo to temp directory
                    repo = git.Repo.clone_from(url, temp_dir)
                    submissions = process_all_submissions(temp_dir, ignore_patterns)

                    for name, code in submissions.items():
                        all_submissions[f"{repo_name}_{name}"] = code

            except Exception as e:
                st.error(f"Error processing {url}: {e}")

    return all_submissions


def calculate_similarity_with_highlights(checker, new_code, threshold=0.3):
    """Calculate similarities and return only significant matches with highlights"""
    scores = checker.calculate_similarities(new_code)

    return scores


def display_comparison_highlights(checker, new_code, similar_submission_name):
    """Display code comparison highlights between new code and similar submission"""
    try:
        # Get the similar submission's code
        similar_code = None
        for name, code in checker.submissions.items():
            if name == similar_submission_name:
                similar_code = code
                break

        if similar_code:
            st.subheader(f"Comparison with: {similar_submission_name}")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**New Submission**")
                st.code(
                    new_code[:2000] + "..." if len(new_code) > 2000 else new_code,
                    language="python",
                )

            with col2:
                st.markdown("**Similar Code Found**")
                st.code(
                    similar_code[:2000] + "..."
                    if len(similar_code) > 2000
                    else similar_code,
                    language="python",
                )

    except Exception as e:
        st.warning(f"Could not display detailed comparison: {e}")


def main():
    st.set_page_config(
        page_title="Copyadi Checker", page_icon="🔍", layout="wide"
    )
    

    st.title("🔍 Copyadi Checker")
    st.markdown("Compare new submissions against batch uploads of previous submissions")

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

    all_previous_submissions = {}
    with tab1:
        zip_file = st.file_uploader(
            "Upload ZIP file containing all previous submissions",
            type=["zip"],
            key="zip_upload",
        )

        if zip_file:
            with st.spinner("Extracting submissions from ZIP..."):
                with tempfile.TemporaryDirectory() as temp_dir:
                    extract_zip(zip_file, temp_dir)
                    extracted_items = os.listdir(temp_dir)
                    extracted_path = os.path.join(temp_dir, extracted_items[0])
                    all_previous_submissions = process_all_submissions(
                        extracted_path, ignore_patterns
                    )

            st.success(
                f"✅ Processed {len(all_previous_submissions)} submissions from ZIP"
            )

    with tab2:
        github_urls_text = st.text_area(
            "Enter GitHub URLs (one per line):",
            placeholder="https://github.com/user/repo1\nhttps://github.com/user/repo2\nhttps://github.com/user/repo3",
            height=150,
        )

        if st.button("Process GitHub URLs", key="process_github"):
            if github_urls_text:
                urls = [
                    url.strip() for url in github_urls_text.split("\n") if url.strip()
                ]
                with st.spinner(f"Processing {len(urls)} GitHub repositories..."):
                    all_previous_submissions = process_github_links(
                        urls, ignore_patterns
                    )
                st.success(f"✅ Processed {len(all_previous_submissions)} submissions")
            else:
                st.warning("Please enter at least one GitHub URL")

    with tab3:
        st.info("Use this for small numbers of individual files")

        prev_submissions = st.file_uploader(
            "Upload previous submissions (ZIP file or multiple files)",
            type=["py", "zip", "java", "cpp", "c", "js", "txt"],
            accept_multiple_files=True,
            key="individual_files",
        )

        if prev_submissions:
            all_previous_submissions = {}
            with tempfile.TemporaryDirectory() as temp_dir:
                prev_dir = Path(temp_dir) / "previous"
                prev_dir.mkdir()

                # Process previous submissions
                for uploaded_file in prev_submissions:
                    if uploaded_file.name.endswith(".zip"):
                        # Extract ZIP file
                        extract_zip(uploaded_file, prev_dir)
                    else:
                        # Save individual file
                        file_path = prev_dir / uploaded_file.name
                        encoding = detect_encoding(uploaded_file)
                        uploaded_file.seek(0)
                        uploaded_code = uploaded_file.read().decode(
                            encoding or "utf-8", errors="ignore"
                        )
                        with open(file_path, "wb") as f:
                            f.write(uploaded_code.encode(encoding or "utf-8"))

                all_previous_submissions = process_all_submissions(
                    str(prev_dir), ignore_patterns
                )

            st.success(f"✅ Loaded {len(all_previous_submissions)} individual files")

    # New submission input
    st.header("🎯 New Submission to Check")
    # new_tab2, new_tab3
    new_tab1,  = st.tabs(
        ["Select File",
         # "Direct Input", "File Upload"
         ]
    )

    with new_tab1:
        if all_previous_submissions:
            new_file_name = None
            file_names = list(all_previous_submissions.keys())
            new_file_name = st.selectbox("select new file:", file_names)
            if new_file_name:
                new_submission_code = all_previous_submissions.pop(new_file_name)

    # with new_tab3:
    #     new_submission_file = st.file_uploader(
    #         "Upload new submission file",
    #         type=["py", "java", "cpp", "c", "js", "txt"],
    #         key="new_submission_file",
    #     )
    #     if new_submission_file:
    #         encoding = detect_encoding(new_submission_file)
    #         new_submission_file.seek(0)  # Reset to beginning
    #         new_submission_code = new_submission_file.read().decode(encoding or "utf-8")
    #
    # with new_tab2:
    #     new_submission_code = st.text_area(
    #         "Or paste code directly:",
    #         height=200,
    #         placeholder="Paste your code here...",
    #         key="direct_code_input",
    #     )

    # Analysis section
    if st.button("🚀 Analyze Similarity", type="primary", use_container_width=True):
        if not all_previous_submissions:
            st.error("❌ Please upload previous submissions first")
            return

        # if not new_submission_code:
        #     st.error("❌ Please provide the new submission to check")
        #     return
        #
        with st.spinner("Analyzing code similarity..."):
            # Initialize checker
            checker = FlexibleCodeSimilarityChecker(
                preprocessing_options=preprocessing_options
            )

            # Add all previous submissions
            for name, code in all_previous_submissions.items():
                checker.add_submission(name, code)

            st.success(
                f"📊 Loaded {len(all_previous_submissions)} submissions for comparison"
            )

            # Calculate similarities
            scores = checker.calculate_similarities(new_submission_code)

            # Display results
            st.header("📈 Results")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Submissions", len(all_previous_submissions))
            with col2:
                st.metric("Similar Submissions", len(scores))
            with col3:
                max_score = max([np.max(r) for r in scores.values()]) if scores else 0
                st.metric("Highest Similarity", f"{max_score:.2%}")

            # Results table
            st.subheader("Similar Submissions Found")
            for i, key in enumerate(scores):
                st.write(f"{i + 1}:  {key} (file: {checker.submission_names[np.argmax(scores[key])]}) = {np.max(scores[key]):.2%}")
                  # display_comparison_highlights(checker, new_submission_code, key)
            #
            # # Warning levels
            # max_similarity = max([r["score"] for r in scores])
            # if max_similarity > 0.8:
            #     st.error(
            #         "🚨 HIGH SIMILARITY - Potential code copying detected!"
            #     )
            # elif max_similarity > 0.6:
            #     st.warning("⚠️ MODERATE SIMILARITY - Review recommended")
            # else:
            #     st.success("✅ Acceptable similarity levels")
            #

    # Batch processing example
    with st.expander("🔄 Advanced: Batch Process Multiple New Submissions"):
        st.markdown("""
        **For processing multiple new submissions at once:**

        1. Upload a ZIP file containing all new submissions to check
        2. The system will compare each against the loaded previous submissions
        3. Get a comprehensive report of all similarities

        *This feature requires additional implementation for batch processing.*
        """)

        if st.button("Implement Batch Processing", disabled=True):
            st.info("Batch processing feature coming soon!")


if __name__ == "__main__":
    main()
