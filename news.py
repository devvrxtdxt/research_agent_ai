# research_agent/news.py
import requests
from phi.agent import Agent

def fetch_trending_articles(api_key, keywords):
    url = "https://newsapi.org/v2/everything"
    articles = []

    for keyword in keywords:
        params = {
            'q': keyword,
            'sortBy': 'popularity',
            'apiKey': api_key,
            'language': 'en',
            'pageSize': 5
        }

        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            for article in data.get('articles', []):
                articles.append({
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'content': article.get('content', ''),
                    'url': article.get('url', '')
                })
        else:
            print(f"Error fetching articles for keyword '{keyword}': {response.status_code}")

    return articles

def generate_social_posts(groq_model, article):
    social_agent = Agent(
        model=groq_model,
        markdown=True,
        description="You are a social media content creator specializing in creating viral, engaging posts.",
        instructions=[
            "Create engaging social media posts from news content",
            "Each post should be unique and platform-appropriate",
            "Include relevant hashtags",
            "Make content engaging and shareable",
            "Keep Twitter posts under 280 characters",
            "Make LinkedIn posts professional and insightful",
            "Make Instagram posts visual and engaging",
            "Always use the exact headers: 'Twitter Post:', 'LinkedIn Post:', and 'Instagram Post:'"
        ]
    )
    
    prompt = f"""
    Based on this news article:
    Title: {article['title']}
    Description: {article['description']}
    
    Create three social media posts with these exact headers:

    Twitter Post:
    [Create a Twitter post here with hashtags, max 280 chars]

    LinkedIn Post:
    [Create a LinkedIn post here with professional tone]

    Instagram Post:
    [Create an Instagram post here with hashtags]

    Make sure to keep the headers exactly as shown above.
    """
    
    response = social_agent.run(prompt)
    return response.content

def generate_content_ideas_from_article(groq_model, article):
    content_agent = Agent(
        model=groq_model,
        markdown=True,
        description="You are a creative content strategist.",
        instructions=[
            "Generate diverse content ideas based on news articles",
            "Create ideas for different platforms and formats",
            "Be specific and actionable",
            "Focus on engaging and viral potential"
        ]
    )
    
    prompt = f"""
    Based on this news article:
    Title: {article['title']}
    Description: {article['description']}
    URL: {article['url']}
    
    Generate 5 creative content ideas that could be created from this news:
    
    For each idea include:
    1. Content format (video, blog, infographic, etc.)
    2. Target platform
    3. Main angle/hook
    4. Key points to cover
    5. Potential hashtags
    
    Make ideas specific and actionable.
    Format in clear markdown with sections.
    """
    
    response = content_agent.run(prompt)
    return response.content

def generate_linkedin_post(groq_model, article, content_idea):
    linkedin_agent = Agent(
        model=groq_model,
        markdown=True,
        description="You are a LinkedIn content expert specializing in creating viral, engaging posts.",
        instructions=[
            "Create professional LinkedIn posts with high engagement potential",
            "Follow the structured format: Hook → Interest Peak → Body → CTA → Hashtags",
            "Make content insightful and valuable",
            "Use line breaks effectively",
            "Include relevant emojis strategically"
        ]
    )
    
    prompt = f"""
    Based on this news article and content idea:
    Article Title: {article['title']}
    Article Description: {article['description']}
    Content Idea: {content_idea}
    
    Create a LinkedIn post with the following structure:
    1. Hook (attention-grabbing first line)
    2. Interest Peak (compelling statement or statistic)
    3. Body (main content with insights, structured in 2-3 paragraphs)
    4. Call-to-Action (engaging CTA)
    5. 3-5 Relevant Hashtags
    
    Use appropriate emojis and line breaks for better readability.
    Format the post in markdown.
    """
    
    response = linkedin_agent.run(prompt)
    return response.content


