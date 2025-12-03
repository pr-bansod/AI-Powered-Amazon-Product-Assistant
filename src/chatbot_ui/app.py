import json
import logging
import uuid

import requests
import streamlit as st
from core.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


st.set_page_config(
    page_title="Ecommerce Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add CSS for blinking animation and improved styling
st.markdown(
    """
<style>
.blinking-status {
    animation: blink 1.5s ease-in-out infinite;
    font-weight: 700;
    color: #ff8c00;
    font-size: 14px;
}

@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    background-color: #ff8c00;
    border-radius: 50%;
    margin-right: 8px;
    animation: pulse 1.5s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.3); opacity: 0.4; }
}

/* Product card styling */
.product-card {
    background-color: #f8f9fa;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 12px;
    border: 1px solid #e0e0e0;
}

/* Product count badge */
.product-count-badge {
    background-color: #ff8c00;
    color: white;
    padding: 6px 12px;
    border-radius: 12px;
    font-weight: 600;
    font-size: 13px;
    display: inline-block;
    margin-bottom: 12px;
}

/* Welcome message styling */
.welcome-message {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
    margin-bottom: 20px;
}

.example-query {
    background-color: rgba(255, 255, 255, 0.1);
    border-radius: 6px;
    padding: 8px 12px;
    margin: 6px 0;
    cursor: pointer;
    font-size: 13px;
}

.example-query:hover {
    background-color: rgba(255, 255, 255, 0.2);
}
</style>
""",
    unsafe_allow_html=True,
)


def get_session_id():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id


session_id = get_session_id()


def api_call(method, url, **kwargs):
    def _show_error_popup(message):
        """Show error message as a popup in the top-right corner."""
        st.session_state["error_popup"] = {
            "visible": True,
            "message": message,
        }

    try:
        response = getattr(requests, method)(url, **kwargs)

        try:
            response_data = response.json()
        except requests.exceptions.JSONDecodeError:
            response_data = {"message": "Invalid response format from server"}

        if response.ok:
            return True, response_data

        return False, response_data

    except requests.exceptions.ConnectionError:
        _show_error_popup("Connection error. Please check your network connection.")
        return False, {"message": "Connection error"}
    except requests.exceptions.Timeout:
        _show_error_popup("The request timed out. Please try again later.")
        return False, {"message": "Request timeout"}
    except Exception as e:
        _show_error_popup(f"An unexpected error occurred: {str(e)}")
        return False, {"message": str(e)}


def api_call_stream(method, url, **kwargs):
    def _show_error_popup(message):
        """Show error message as a popup in the top-right corner."""
        st.session_state["error_popup"] = {
            "visible": True,
            "message": message,
        }

    try:
        response = getattr(requests, method)(url, **kwargs)

        return response.iter_lines()

    except requests.exceptions.ConnectionError:
        _show_error_popup("Connection error. Please check your network connection.")
        return False, {"message": "Connection error"}
    except requests.exceptions.Timeout:
        _show_error_popup("The request timed out. Please try again later.")
        return False, {"message": "Request timeout"}
    except Exception as e:
        _show_error_popup(f"An unexpected error occurred: {str(e)}")
        return False, {"message": str(e)}


def submit_feedback(feedback_type=None, feedback_text=""):
    """Submit feedback to the API endpoint"""

    def _feedback_score(feedback_type):
        if feedback_type == "positive":
            return 1
        elif feedback_type == "negative":
            return 0
        else:
            return None

    feedback_data = {
        "feedback_score": _feedback_score(feedback_type),
        "feedback_text": feedback_text,
        "trace_id": st.session_state.trace_id,
        "thread_id": session_id,
        "feedback_source_type": "api",
    }

    logger.info(f"Feedback data: {feedback_data}")

    status, response = api_call("post", f"{config.API_URL}/submit_feedback", json=feedback_data)
    return status, response


if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]

if "used_context" not in st.session_state:
    st.session_state.used_context = []

if "shopping_cart" not in st.session_state:
    st.session_state.shopping_cart = []

# Initialize feedback states (simplified)
if "latest_feedback" not in st.session_state:
    st.session_state.latest_feedback = None

if "show_feedback_box" not in st.session_state:
    st.session_state.show_feedback_box = False

if "feedback_submission_status" not in st.session_state:
    st.session_state.feedback_submission_status = None

if "trace_id" not in st.session_state:
    st.session_state.trace_id = None


with st.sidebar:
    # Create tabs in the sidebar
    suggestions_tab, cart_tab = st.tabs(["🔍 Suggestions", "🛒 Shopping Cart"])

    # Suggestions Tab
    with suggestions_tab:
        if st.session_state.used_context:
            for idx, item in enumerate(st.session_state.used_context):
                st.caption(item.get("description", "No description"))
                if "image_url" in item:
                    st.image(item["image_url"], width=250)
                st.caption(f"Price: {item['price']} USD")
                st.divider()
        else:
            st.info("No suggestions yet")

    # Shopping Cart Tab
    with cart_tab:
        if st.session_state.shopping_cart:
            total_price = 0
            for idx, item in enumerate(st.session_state.shopping_cart):
                if "product_image_url" in item and item["product_image_url"]:
                    st.image(item["product_image_url"], width=250)
                st.caption(f"Quantity: {item.get('quantity', 0)}")
                st.caption(f"Price: {item.get('price', 0)} {item.get('currency', 'USD')}")
                st.caption(f"Total: {item.get('total_price', 0)} {item.get('currency', 'USD')}")
                if item["total_price"] : 
                    total_price += float(item.get('total_price', 0))
                    st.divider()
                else:
                    continue    
            st.markdown(f"**Cart Total: {total_price:.2f} USD**")
        else:
            st.info("Your shopping cart is empty")


# Show welcome message with example queries (only when no conversation started)
if len(st.session_state.messages) == 1:
    st.markdown(
        """
        <div class="welcome-message">
            <h3 style="margin-top: 0;">👋 Welcome to the E-Commerce Assistant!</h3>
            <p style="margin-bottom: 12px;">I can help you find products and answer questions. Try asking:</p>
            <div class="example-query">🎧 "Show me the best noise-canceling headphones under $200"</div>
            <div class="example-query">📱 "What are the top-rated smartphones with good battery life?"</div>
            <div class="example-query">⌚ "can you give me a smartwatch compatible with iOS. Also give me good and bad reviews for each item."</div>
            <div class="example-query">💻 "can you give me a smartwatch compatible with iOS. Also give me good and bad reviews for each item. Add the most durable product to the cart"</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Add feedback buttons only for the latest assistant message (excluding the initial greeting)
        is_latest_assistant = message["role"] == "assistant" and idx == len(st.session_state.messages) - 1 and idx > 0

        if is_latest_assistant:
            # Use Streamlit's built-in feedback component
            feedback_key = f"feedback_{len(st.session_state.messages)}"
            feedback_result = st.feedback("thumbs", key=feedback_key)

            # Handle feedback selection
            if feedback_result is not None:
                feedback_type = "positive" if feedback_result == 1 else "negative"

                # Only submit if this is a new/different feedback
                if st.session_state.latest_feedback != feedback_type:
                    with st.spinner("Submitting feedback..."):
                        status, response = submit_feedback(feedback_type=feedback_type)
                        if status:
                            st.session_state.latest_feedback = feedback_type
                            st.session_state.feedback_submission_status = "success"
                            st.session_state.show_feedback_box = feedback_type == "negative"
                        else:
                            st.session_state.feedback_submission_status = "error"
                            st.error("Failed to submit feedback. Please try again.")
                    st.rerun()

            # Show feedback status message
            if st.session_state.latest_feedback and st.session_state.feedback_submission_status == "success":
                if st.session_state.latest_feedback == "positive":
                    st.success("✅ Thank you for your positive feedback!")
                elif st.session_state.latest_feedback == "negative" and not st.session_state.show_feedback_box:
                    st.success("✅ Thank you for your feedback!")
            elif st.session_state.feedback_submission_status == "error":
                st.error("❌ Failed to submit feedback. Please try again.")

            # Show feedback text box if thumbs down was pressed
            if st.session_state.show_feedback_box:
                st.markdown("**Want to tell us more? (Optional)**")
                st.caption(
                    "Your negative feedback has already been recorded. You can optionally provide additional details below."
                )

                # Text area for detailed feedback
                feedback_text = st.text_area(
                    "Additional feedback (optional)",
                    key=f"feedback_text_{len(st.session_state.messages)}",
                    placeholder="Please describe what was wrong with this response...",
                    height=100,
                )

                # Send additional feedback button
                col_send, col_spacer, col_close = st.columns([3, 5, 2])
                with col_send:
                    if st.button("Send Additional Details", key=f"send_additional_{len(st.session_state.messages)}"):
                        if feedback_text.strip():  # Only send if there's actual text
                            with st.spinner("Submitting additional feedback..."):
                                status, response = submit_feedback(feedback_text=feedback_text)
                                if status:
                                    st.success("✅ Thank you! Your additional feedback has been recorded.")
                                    st.session_state.show_feedback_box = False
                                else:
                                    st.error("❌ Failed to submit additional feedback. Please try again.")
                        else:
                            st.warning("Please enter some feedback text before submitting.")
                        st.rerun()

                with col_close:
                    if st.button("Close", key=f"close_feedback_{len(st.session_state.messages)}"):
                        st.session_state.show_feedback_box = False
                        st.rerun()


if prompt := st.chat_input("Hello! How can I assist you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        message_placeholder = st.empty()

        for line in api_call_stream(
            "post",
            f"{config.API_URL}/rag",
            json={"query": prompt, "thread_id": session_id},
            stream=True,
            headers={"Accept": "text/event-stream"},
        ):
            # Skip empty lines or non-bytes responses
            if not line or not isinstance(line, bytes):
                continue

            line_text = line.decode("utf-8")

            if line_text.startswith("data: "):
                data = line_text[6:]

                try:
                    output = json.loads(data)

                    if output["type"] == "final_result":
                        answer = output["data"]["answer"]
                        used_context = output["data"]["used_context"]
                        trace_id = output["data"]["trace_id"]
                        shopping_cart = output["data"].get("shopping_cart", [])

                        st.session_state.used_context = used_context
                        st.session_state.shopping_cart = shopping_cart
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        st.session_state.trace_id = trace_id

                        st.session_state.latest_feedback = None
                        st.session_state.show_feedback_box = False
                        st.session_state.feedback_submission_status = None

                        status_placeholder.empty()
                        message_placeholder.markdown(answer)
                        break

                except json.JSONDecodeError as e:
                    # Show status with blinking animation
                    status_placeholder.markdown(
                        f'<div class="blinking-status"><span class="status-dot"></span>{data}</div>', unsafe_allow_html=True
                    )

    st.rerun()
