import streamlit as st
import asyncio
import sys
from pathlib import Path
import io
import pandas as pd

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from graph import build_graph
from voice_processor import process_voice_input, generate_voice_response
from comparison_table import create_comparison_table, format_table_markdown, format_table_text

# Page config
st.set_page_config(
    page_title="Voice-to-Voice Product Recommendation",
    page_icon="🎤",
    layout="wide"
)

st.title("🎤 Voice-to-Voice Product Recommendation")
st.markdown("Record your product query and get recommendations with audio response!")

# Initialize session state
if "audio_response" not in st.session_state:
    st.session_state.audio_response = None
if "comparison_table" not in st.session_state:
    st.session_state.comparison_table = None
if "final_answer" not in st.session_state:
    st.session_state.final_answer = None

# Sidebar for instructions
with st.sidebar:
    st.header("Instructions")
    st.markdown("""
    1. Click "Start Recording" to record your query
    2. Speak your product request (e.g., "Find eco-friendly cleaners under $20")
    3. Click "Stop Recording" when done
    4. The system will process your query and show results
    5. Listen to the audio response
    """)
    
    st.header("Example Queries")
    st.markdown("""
    - "I want to buy Barbie dolls."
    - "What is Catan the board game?"
    - "I want to buy a skate board under 300 dollars."
    """)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Record Your Query")
    
    # Audio recorder
    audio_file = st.audio_input("Record your product query")
    
    if audio_file:
        st.success("Audio recorded! Processing...")
        
        # Read audio bytes from UploadedFile
        audio_bytes = audio_file.read()
        
        # Process audio to text
        with st.spinner("Converting speech to text..."):
            try:
                user_query = process_voice_input(audio_bytes)
                st.text_area("Transcribed Query:", user_query, height=100)
            except Exception as e:
                st.error(f"Error processing audio: {e}")
                import traceback
                st.code(traceback.format_exc())
                user_query = None
        
        if user_query:
            # Run pipeline
            with st.spinner("Processing your query through the agent pipeline..."):
                try:
                    workflow = build_graph()
                    # Run async workflow - handle event loop properly for Streamlit
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    result = loop.run_until_complete(
                        workflow.ainvoke({"user_input": user_query})
                    )
                    
                    # Extract results
                    final_answer = result.get("final_answer", "")
                    retrieval_results = result.get("retriever", {}).get("retrieval_results", {})
                    results = retrieval_results.get("results", [])
                    
                    # Store in session state
                    st.session_state.final_answer = final_answer
                    st.session_state.retrieval_results = results
                    
                    # Create comparison table
                    if results:
                        comparison_df = create_comparison_table(results)
                        st.session_state.comparison_table = comparison_df
                    
                    # Automatically generate audio response
                    with st.spinner("Generating audio response..."):
                        try:
                            # Use the final answer for audio (full recommendation)
                            audio_text = final_answer
                            audio_bytes = generate_voice_response(audio_text)
                            st.session_state.audio_response = audio_bytes
                        except Exception as e:
                            st.warning(f"Could not generate audio: {e}")
                            st.session_state.audio_response = None
                    
                    st.success("Processing complete!")
                    
                except Exception as e:
                    st.error(f"Error running pipeline: {e}")
                    import traceback
                    st.code(traceback.format_exc())

with col2:
    st.header("Results")
    
    # Show final answer
    if st.session_state.final_answer:
        st.subheader("Recommendation")
        st.markdown(st.session_state.final_answer)
        
        # Play audio if available (auto-generated)
        if st.session_state.audio_response:
            st.subheader("Audio Response")
            st.audio(st.session_state.audio_response, format="audio/mp3", autoplay=False)
            st.download_button(
                label="Download Audio",
                data=st.session_state.audio_response,
                file_name="product_recommendation.mp3",
                mime="audio/mp3"
            )
            
            # Optional: Regenerate audio button
            if st.button("Regenerate Audio"):
                with st.spinner("Regenerating audio..."):
                    try:
                        audio_text = st.session_state.final_answer
                        audio_bytes = generate_voice_response(audio_text)
                        st.session_state.audio_response = audio_bytes
                        st.success("Audio regenerated!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error regenerating audio: {e}")
        else:
            st.info("Audio response will be generated automatically when results are ready.")
    
    # Show comparison table
    if st.session_state.comparison_table is not None:
        st.subheader("📋 Product Comparison Table")
        st.dataframe(st.session_state.comparison_table, use_container_width=True)
        
        # Download table as CSV
        csv = st.session_state.comparison_table.to_csv(index=False)
        st.download_button(
            label="Download Table as CSV",
            data=csv,
            file_name="product_comparison.csv",
            mime="text/csv"
        )
        
        # Show markdown version
        with st.expander("View as Markdown"):
            st.markdown(format_table_markdown(st.session_state.comparison_table))

# Footer
st.markdown("---")
st.markdown("**Powered by:** LangGraph Agentic RAG | Whisper STT | OpenAI TTS")

