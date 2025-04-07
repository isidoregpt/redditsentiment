import streamlit as st
import os
import praw
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import zipfile
import io

# Download the VADER lexicon (comment this out if already downloaded)
nltk.download('vader_lexicon')

st.title("Reddit Sentiment Analysis")

# --- Input Section ---
st.header("PRAW Credentials")
client_id = st.text_input("Client ID")
client_secret = st.text_input("Client Secret", type="password")
user_agent = st.text_input("User Agent")

st.header("Subreddits (Enter up to 10, separated by commas)")
subreddits_input = st.text_area("Subreddits", placeholder="e.g., python, dataisbeautiful")

st.header("Keywords (Enter up to 10, separated by commas)")
keywords_input = st.text_area("Keywords", placeholder="e.g., streamlit, analysis")

if st.button("Run Analysis"):
    # Validate credentials
    if not client_id or not client_secret or not user_agent:
        st.error("Please enter all PRAW credentials.")
    else:
        # Parse subreddits and keywords into lists
        subreddits = [s.strip() for s in subreddits_input.split(",") if s.strip()]
        keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]
        
        if not subreddits:
            st.error("Please enter at least one subreddit.")
        elif not keywords:
            st.error("Please enter at least one keyword.")
        else:
            # Initialize PRAW
            try:
                reddit = praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent=user_agent
                )
            except Exception as e:
                st.error(f"Failed to initialize PRAW:\n{e}")
            else:
                st.info("Scraping Reddit posts and comments... This may take some time.")
                all_data = []
                
                # Process each subreddit
                for subreddit_name in subreddits:
                    try:
                        # Limit to 50 new posts per subreddit (adjust as needed)
                        all_posts = reddit.subreddit(subreddit_name).new(limit=50)
                        for post in all_posts:
                            title_lower = post.title.lower()
                            if any(k.lower() in title_lower for k in keywords):
                                # Expand all comments (removing MoreComments objects)
                                post.comments.replace_more(limit=0)
                                for comment in post.comments.list():
                                    all_data.append({
                                        "text": comment.body,
                                        "title": post.title,
                                        "url": post.url,
                                        "subreddit": subreddit_name,
                                        "date": datetime.fromtimestamp(comment.created_utc)
                                    })
                    except Exception as e:
                        st.write(f"Error processing subreddit '{subreddit_name}': {e}")

                if not all_data:
                    st.info("No comments were found matching the specified keywords.")
                else:
                    # Create DataFrame and reorder columns
                    df = pd.DataFrame(all_data)
                    df = df[["text", "title", "url", "subreddit", "date"]]
                    
                    # Create an output folder with a timestamp
                    now = datetime.now()
                    dt_string = now.strftime("%m_%d_%Y_%H_%M")
                    folder_name = f"Data_{dt_string}"
                    os.makedirs(folder_name, exist_ok=True)
                    
                    # Save CSV and TAB files
                    csv_path = os.path.join(folder_name, f"{folder_name}.csv")
                    tab_path = os.path.join(folder_name, f"{folder_name}.tab")
                    df.to_csv(csv_path, index=False)
                    df.to_csv(tab_path, sep="\t", index=False)
                    
                    # --- Sentiment Analysis ---
                    st.info("Performing sentiment analysis...")
                    sid = SentimentIntensityAnalyzer()
                    df['scores'] = df['text'].apply(lambda x: sid.polarity_scores(str(x)))
                    df['compound'] = df['scores'].apply(lambda x: x['compound'])
                    df['neg'] = df['scores'].apply(lambda x: x['neg'])
                    df['neu'] = df['scores'].apply(lambda x: x['neu'])
                    df['pos'] = df['scores'].apply(lambda x: x['pos'])
                    
                    # Function to classify sentiment based on compound score
                    def classify_sentiment(score):
                        if score >= 0.05:
                            return 'positive'
                        elif score <= -0.05:
                            return 'negative'
                        else:
                            return 'neutral'
                    
                    df['sentiment'] = df['compound'].apply(classify_sentiment)
                    
                    # Save sentiment analysis results
                    sentiment_csv = os.path.join(folder_name, "sentiment_analysis_results.csv")
                    df.to_csv(sentiment_csv, index=False)
                    
                    # --- Display Summary ---
                    sentiment_counts = df['sentiment'].value_counts()
                    total_posts = len(df)
                    pos_count = sentiment_counts.get('positive', 0)
                    neg_count = sentiment_counts.get('negative', 0)
                    neu_count = sentiment_counts.get('neutral', 0)
                    
                    st.success("Sentiment Analysis Complete!")
                    st.write(f"**Total Comments Analyzed:** {total_posts}")
                    st.write(f"**Positive:** {pos_count}")
                    st.write(f"**Negative:** {neg_count}")
                    st.write(f"**Neutral:** {neu_count}")
                    st.write(f"**Results saved in folder:** {folder_name}")
                    
                    # --- Plotting ---
                    # Time-series plot of sentiment over time
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    sentiment_over_time = df.resample('D')['sentiment'].value_counts().unstack().fillna(0)
                    
                    fig1, ax1 = plt.subplots(figsize=(10, 5))
                    for s in ['positive', 'neutral', 'negative']:
                        if s in sentiment_over_time.columns:
                            ax1.plot(sentiment_over_time.index, sentiment_over_time[s], label=s)
                    ax1.set_title("Sentiment Over Time")
                    ax1.set_xlabel("Date")
                    ax1.set_ylabel("Number of Comments")
                    ax1.legend()
                    plt.tight_layout()
                    
                    # Save and display the time-series plot
                    time_plot_path = os.path.join(folder_name, "sentiment_over_time.png")
                    plt.savefig(time_plot_path)
                    st.pyplot(fig1)
                    plt.close(fig1)
                    
                    # Bar chart for overall sentiment distribution
                    fig2, ax2 = plt.subplots(figsize=(6, 4))
                    sentiment_counts.plot(kind='bar', color=['green', 'gray', 'red'], ax=ax2)
                    ax2.set_title("Overall Sentiment Distribution")
                    ax2.set_xlabel("Sentiment")
                    ax2.set_ylabel("Count")
                    plt.tight_layout()
                    
                    # Save and display the bar chart
                    dist_plot_path = os.path.join(folder_name, "sentiment_distribution.png")
                    plt.savefig(dist_plot_path)
                    st.pyplot(fig2)
                    plt.close(fig2)
                    
                    # --- Create In-Memory ZIP File for Download ---
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                        output_files = [
                            csv_path,
                            tab_path,
                            sentiment_csv,
                            time_plot_path,
                            dist_plot_path
                        ]
                        for file_path in output_files:
                            zip_file.write(file_path, arcname=os.path.basename(file_path))
                    
                    zip_buffer.seek(0)
                    
                    st.download_button(
                        label="Download All Output Files as ZIP",
                        data=zip_buffer,
                        file_name=f"{folder_name}.zip",
                        mime="application/zip"
                    )
                    
                    if st.checkbox("Show Raw Data"):
                        st.dataframe(df.reset_index())
