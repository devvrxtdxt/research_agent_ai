import os
from dotenv import load_dotenv
import streamlit as st
from phi.agent import Agent, RunResponse
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.googlesearch import GoogleSearch
from phi.model.groq import Groq

# Load environment variables from .env file
load_dotenv()

# Set up API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Validate API key
if not GROQ_API_KEY:
    raise ValueError("Missing required GROQ API key. Please check your .env file.")

# Initialize Groq model
groq_model = Groq(
    id="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY
)

def create_search_agents():
    # DuckDuckGo Agent
    ddg_agent = Agent(
        tools=[DuckDuckGo()],
        model=groq_model,
        markdown=True,
        description="You are a search agent that helps users find information using DuckDuckGo.",
        instructions=[
            "When searching, return results in a structured format.",
            "Each result should include a title, link, and snippet.",
            "Focus on providing accurate and relevant information.",
            "If possible, return results as a list of dictionaries."
        ],
        show_tool_calls=True
    )
    
    # Google Search Agent
    google_agent = Agent(
        tools=[GoogleSearch()],
        model=groq_model,
        markdown=True,
        description="You are a search agent that helps users find information using Google.",
        instructions=[
            "When searching, return results in a structured format.",
            "Each result should include a title, link, and snippet.",
            "Focus on providing accurate and relevant information.",
            "If possible, return results as a list of dictionaries."
        ],
        show_tool_calls=True
    )
    
    return ddg_agent, google_agent

def search_with_agent(agent, keyword) -> list:
    try:
        run: RunResponse = agent.run(f"Find detailed information about: {keyword}")
        
        # Check if run.content is a string (direct response)
        if isinstance(run.content, str):
            # Parse the response text to extract information
            return [{
                'title': 'Search Result',
                'link': '',
                'snippet': run.content
            }]
        
        # If it's a dictionary with results
        elif isinstance(run.content, dict) and 'results' in run.content:
            return [{
                'title': item.get('title', 'No title'),
                'link': item.get('link', ''),
                'snippet': item.get('snippet', '')
            } for item in run.content['results']]
        
        # If it's a list of results
        elif isinstance(run.content, list):
            return [{
                'title': item.get('title', 'No title'),
                'link': item.get('link', ''),
                'snippet': item.get('snippet', '')
            } for item in run.content]
        
        else:
            return [{
                'title': 'Search Result',
                'link': '',
                'snippet': str(run.content)
            }]
            
    except Exception as e:
        st.error(f"Search failed: {str(e)}")
        return []

def generate_content_ideas(keywords):
    try:
        content_agent = Agent(
            model=groq_model,
            markdown=True,
            description="You are a creative content idea generator.",
            instructions=[
                "Generate engaging and creative content ideas based on keywords.",
                "Format the output in clear markdown.",
                "Be specific and actionable in your suggestions."
            ]
        )
        
        prompt = f"""Given these keywords: {', '.join(keywords)}
        Generate 5 content ideas that would be interesting and engaging.
        For each idea, provide:
        1. A catchy title
        2. A brief description
        3. Key points to cover
        
        Format the output in markdown."""
        
        run: RunResponse = content_agent.run(prompt)
        return run.content
    except Exception as e:
        st.error(f"Content generation failed: {str(e)}")
        return "Failed to generate content ideas."

# Set up Streamlit page config
st.set_page_config(
    page_title="Research Agent",
    page_icon="🔍",
    layout="wide"
)

# Initialize agents
ddg_agent, google_agent = create_search_agents()

# Add title and description to the Streamlit app
st.title("Research Agent")
st.write("Enter keywords to search across multiple platforms and generate content ideas.")

# Create input field for keywords
keywords_input = st.text_area("Enter keywords (one per line)", height=100)
max_results = st.slider("Maximum results per search engine", min_value=1, max_value=10, value=5)

if st.button("Start Research"):
    if keywords_input:
        keywords = [k.strip() for k in keywords_input.split('\n') if k.strip()]
        
        with st.spinner('Searching and generating content ideas...'):
            # Create tabs for different results
            search_tab, ideas_tab = st.tabs(["Search Results", "Content Ideas"])
            
            # Perform searches
            search_results = []
            for keyword in keywords:
                with st.status(f"Searching for: {keyword}") as status:
                    st.write("Searching DuckDuckGo...")
                    ddg_results = search_with_agent(ddg_agent, keyword)
                    search_results.extend(ddg_results[:max_results])
                    
                    st.write("Searching Google...")
                    google_results = search_with_agent(google_agent, keyword)
                    search_results.extend(google_results[:max_results])
                    
                    status.update(label="Search completed!", state="complete")

            # Display search results in the first tab
            with search_tab:
                if not search_results:
                    st.warning("No results found.")
                else:
                    for idx, result in enumerate(search_results, 1):
                        with st.container():
                            st.markdown(f"### Result {idx}")
                            title = result.get('title', 'No title')
                            link = result.get('link', '')
                            snippet = result.get('snippet', '')
                            
                            if title and title != 'No title':
                                st.markdown(f"**{title}**")
                            if link:
                                st.markdown(f"🔗 [{link}]({link})")
                            if snippet:
                                st.markdown(f"_{snippet}_")
                            st.divider()

            # Generate and display content ideas in the second tab
            with ideas_tab:
                content_ideas = generate_content_ideas(keywords)
                st.markdown(content_ideas)
    else:
        st.error("Please enter at least one keyword")

# Add footer with information
st.markdown("---")
st.markdown("Built with Streamlit, Phi Framework, and Groq")