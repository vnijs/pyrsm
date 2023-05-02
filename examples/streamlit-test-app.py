import streamlit as st

st.markdown(
    """
    <div id="my-input">
        <label for="input-field">Enter a value:</label>
        <input id="input-field" type="text">
        <button onclick="sendValue()">Submit</button>
    </div>

    <script>
        function sendValue() {
            const inputValue = document.getElementById('input-field').value;
            const event = new CustomEvent('customEvent', { detail: inputValue });
            document.dispatchEvent(event);
        }
    </script>
    """,
    unsafe_allow_html=True,
)

input_value = st.empty()


@st.on_event("customEvent")
def handle_custom_event(event):
    input_value.markdown(f"You entered: **{event.detail}**")
