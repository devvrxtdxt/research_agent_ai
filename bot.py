# research_agent/bot.py
import os
from dotenv import load_dotenv
import streamlit as st
from phi.agent import Agent, RunResponse
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.groq import Groq
import requests
from news import (
    fetch_trending_articles, 
    generate_social_posts, 
    generate_content_ideas_from_article,
    generate_linkedin_post
)

# Rest of your bot.py code remains the same...

# Load environment variables from .env file
load_dotenv()
# Set up API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")  # Add your News API key

# Validate API keys
if not GROQ_API_KEY:
    raise ValueError("Missing required GROQ API key. Please check your .env file.")
if not NEWS_API_KEY:
    raise ValueError("Missing required News API key. Please check your .env file.")

# Initialize Groq model
groq_model = Groq(
    id="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY
)

def create_search_agent():
    # DuckDuckGo Agent with news functionality
    ddg_agent = Agent(
        tools=[DuckDuckGo()],
        model=groq_model,
        markdown=True,
        description="You are a search agent that helps users find information and news using DuckDuckGo.",
        instructions=[
            "When searching, return results in a structured format.",
            "Each result should include a title, link, and snippet.",
            "Focus on providing accurate and relevant information.",
            "If possible, return results as a list of dictionaries."
        ],
        show_tool_calls=True
    )
    
    return ddg_agent

def search_with_agent(agent, keyword) -> list:
    try:
        run: RunResponse = agent.run(f"Find detailed information and news about: {keyword}")
        
        # Check if run.content is a string (direct response)
        if isinstance(run.content, str):
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

def fetch_news_articles(keywords):
    articles = []
    for keyword in keywords:
        articles.extend(fetch_trending_articles(NEWS_API_KEY, [keyword.strip()]))
    return articles

# Set up Streamlit page config
st.set_page_config(
    page_title="Research Agent",
    page_icon="🔍",
    layout="wide"
)

# Initialize agent
ddg_agent = create_search_agent()

# Add title and description to the Streamlit app
st.title("Research Agent")
st.write("Enter keywords to search for information and news, and generate content ideas.")

# Create input field for keywords
keywords_input = st.text_area("Enter keywords (one per line)", height=100)
max_results = st.slider("Maximum results", min_value=1, max_value=10, value=5)

if st.button("Start Research"):
    if keywords_input:
        keywords = [k.strip() for k in keywords_input.split('\n') if k.strip()]
        
        with st.spinner('Searching and generating content ideas...'):
            # Create tabs for different results
            search_tab, ideas_tab, news_tab = st.tabs(["Search Results", "Content Ideas", "News Articles"])
            
            # Perform searches
            search_results = []
            for keyword in keywords:
                with st.status(f"Searching for: {keyword}") as status:
                    st.write("Searching DuckDuckGo for information and news...")
                    ddg_results = search_with_agent(ddg_agent, keyword)
                    search_results.extend(ddg_results[:max_results])
                    
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
                                st.markdown(f"{title}")
                            if link:
                                st.markdown(f"🔗 [{link}]({link})")
                            if snippet:
                                st.markdown(f"{snippet}")
                            st.divider()

            # Generate and display content ideas in the second tab
            with ideas_tab:
                content_ideas = generate_content_ideas(keywords)
                st.markdown(content_ideas)

            # Fetch and display news articles in the third tab
            with news_tab:
                news_articles = fetch_news_articles(keywords)
                if not news_articles:
                    st.warning("No news articles found.")
                else:
                    for idx, article in enumerate(news_articles[:max_results], 1):
                        with st.expander(f"📰 Article {idx}: {article['title']}", expanded=True):
                            tabs = st.tabs(["Social Media Posts", "Content Ideas", "LinkedIn Posts"])
                            
                            # Social Media Posts Tab
                            with tabs[0]:
                                social_content = generate_social_posts(groq_model, article)
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.markdown("### 🐦 Twitter")
                                    # More robust content splitting
                                    try:
                                        twitter_content = social_content.split("Twitter Post:")[1].split("LinkedIn Post:")[0].strip()
                                    except IndexError:
                                        twitter_content = "Error generating Twitter content"
                                    st.markdown(twitter_content)
                                    st.download_button(
                                        "Copy Twitter Post",
                                        twitter_content,
                                        key=f"twitter_{idx}"
                                    )
                                
                                with col2:
                                    st.markdown("### 💼 LinkedIn")
                                    try:
                                        linkedin_content = social_content.split("LinkedIn Post:")[1].split("Instagram Post:")[0].strip()
                                    except IndexError:
                                        linkedin_content = "Error generating LinkedIn content"
                                    st.markdown(linkedin_content)
                                    st.download_button(
                                        "Copy LinkedIn Post",
                                        linkedin_content,
                                        key=f"linkedin_{idx}"
                                    )
                                
                                with col3:
                                    st.markdown("### 📸 Instagram")
                                    try:
                                        instagram_content = social_content.split("Instagram Post:")[1].strip()
                                    except IndexError:
                                        instagram_content = "Error generating Instagram content"
                                    st.markdown(instagram_content)
                                    st.download_button(
                                        "Copy Instagram Post",
                                        instagram_content,
                                        key=f"instagram_{idx}"
                                    )
                            
                            # Content Ideas Tab
                            with tabs[1]:
                                st.markdown("### 🎯 Content Ideas From This Article")
                                article_ideas = generate_content_ideas_from_article(groq_model, article)
                                st.markdown(article_ideas)
                                st.download_button(
                                    "Copy All Content Ideas",
                                    article_ideas,
                                    key=f"ideas_{idx}"
                                )

                            # LinkedIn Posts Tab (New Main Section)
                            with tabs[2]:
                                st.markdown("### 💼 Professional LinkedIn Posts")
                                
                                # Generate LinkedIn post from article
                                main_linkedin_post = generate_linkedin_post(groq_model, article, "Main article theme")
                                
                                st.markdown("#### 📌 Main Article Post")
                                
                                # Create columns for main post and structure
                                main_post_col, main_structure_col = st.columns([2, 1])
                                
                                with main_post_col:
                                    st.markdown(main_linkedin_post)
                                    st.download_button(
                                        "Copy Main LinkedIn Post",
                                        main_linkedin_post,
                                        key=f"main_linkedin_{idx}"
                                    )
                                
                                with main_structure_col:
                                    st.markdown("**Post Structure:**")
                                    st.markdown("""
                                    - 🎣 Hook
                                    - 📈 Interest Peak
                                    - 📝 Body
                                    - 🎯 CTA
                                    - #️⃣ Hashtags
                                    """)
                                
                                st.markdown("---")
                                st.markdown("#### 🔄 Alternative Angle Posts")
                                
                                # Generate 2-3 alternative LinkedIn posts with different angles
                                angles = [
                                    "Industry impact and trends",
                                    "Problem-solution perspective",
                                    "Future implications"
                                ]
                                
                                # Create tabs for alternative posts instead of expanders
                                alt_tabs = st.tabs([f"Alternative {i+1}: {angle}" for i, angle in enumerate(angles)])
                                
                                for i, (angle, tab) in enumerate(zip(angles, alt_tabs)):
                                    with tab:
                                        alt_linkedin_post = generate_linkedin_post(groq_model, article, angle)
                                        
                                        # Create columns for post and structure
                                        post_col, structure_col = st.columns([2, 1])
                                        
                                        with post_col:
                                            st.markdown(alt_linkedin_post)
                                            st.download_button(
                                                "Copy This Version",
                                                alt_linkedin_post,
                                                key=f"linkedin_alt_{idx}_{i}"
                                            )
                                        
                                        with structure_col:
                                            st.markdown("**Post Structure:**")
                                            st.markdown("""
                                            - 🎣 Hook
                                            - 📈 Interest Peak
                                            - 📝 Body
                                            - 🎯 CTA
                                            - #️⃣ Hashtags
                                            """)
                            
                            st.divider()
    else:
        st.error("Please enter at least one keyword")

# Add footer with information
st.markdown("---")
st.markdown("Built with Streamlit, Phi Framework, and Groq")


