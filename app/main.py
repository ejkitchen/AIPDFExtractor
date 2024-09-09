import streamlit as st
from pathlib import Path
import tempfile
import logging
from unidecode import unidecode

from utils import get_torch_gpu_info

from docling.document_converter import DocumentConverter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s -> %(levelname)s -> %(name)s -> %(module)s:%(lineno)d -> %(message)s",
    datefmt="%a %b %d: %H:%M:%S",
    handlers=[
        logging.StreamHandler()
    ]
)

def perform_pdf_extraction(file_path: str) -> str:
    try:
        logging.info(f"Performing PDF extraction for {file_path}")
        logging.info(f"Loading DocumentConverter...")
        converter = DocumentConverter()
        logging.info(f"DocumentConverter loaded successfully")
        logging.info(f"Converting document...")
        doc = converter.convert_single(file_path)
        logging.info(f"Document converted successfully")
        
        logging.info(f"Rendering document as markdown...")
        output = doc.render_as_markdown()
        logging.info(f"Output: {output}")

        logging.info(f"Unidecoding output...")
        output = unidecode(output)
        logging.info(f"Output unidecoded successfully")
        
        return output
    
    except Exception as e:
        logging.error(f"Error during PDF extraction: {str(e)}")
        raise

def show():
    st.set_page_config(page_title="AI PDF Extractor", page_icon="📄")
    
    st.title("AI PDF Extractor")
    st.write("Upload a PDF file to extract its content using AI.")

    # GPU Info Expander
    with st.expander("GPU Information"):
        gpu_info = get_torch_gpu_info()
        if isinstance(gpu_info, str):
            st.write(gpu_info)
        else:
            for line in gpu_info:
                if line.startswith("\nGPU") or line.startswith("\nSystem Memory:"):
                    st.subheader(line.strip())
                elif line.startswith("  "):
                    st.text(line)
                else:
                    st.write(line)

    uploaded_file = st.file_uploader("Upload your PDF file here", type=["pdf"])

    if uploaded_file:
        st.success("File uploaded successfully!")

        if st.button("Extract Content"):
            with st.spinner("Extracting content..."):
                # Save the uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                try:
                    extracted_text = perform_pdf_extraction(tmp_file_path)
                    st.markdown("### Extracted Content")
                    st.markdown(extracted_text)
                    
                    # Offer download option
                    st.download_button(
                        label="Download Extracted Text",
                        data=extracted_text,
                        file_name="extracted_content.txt",
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error(f"An error occurred during extraction: {str(e)}")
                finally:
                    # Clean up the temporary file
                    Path(tmp_file_path).unlink()
    else:
        st.info("Please upload a PDF file to begin.")

if __name__ == "__main__":
    show()