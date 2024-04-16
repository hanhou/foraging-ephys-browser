import streamlit as st

st.set_page_config(layout="centered")

def app():
    st.markdown('## Dear Foraging Ephys Browser user, \n'
                '#### For better stability and accessibility, this Ephys app has been migrated to Amazon ECS (thank you Yosef and Jon 🙌).\n'
                '#### 👉  [foraging-ephys-browser.allenneuraldynamics-test.org](https://foraging-ephys-browser.allenneuraldynamics-test.org/) '
                )
    st.markdown('#### If you see a window asking you to "Select a certificate", please click "Cancel".')
    st.divider()
    st.markdown('Ask Han or submit an issue [here](https://github.com/AllenNeuralDynamics/foraging-ephys-browser/issues) if you have any questions.')

app()

