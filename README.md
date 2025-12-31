Sleep Health & Lifestyle Dashboard

Health-focused data analytics project built with Python and Flask

This project explores relationships between lifestyle factors (physical activity, stress, occupation) and sleep outcomes using an interactive web dashboard. It was built as a portfolio project to demonstrate applied data analysis, visualization, and basic modeling in a health context.

Project Overview: 
- Built a Flask web application to analyze and visualize sleep health data
- Designed interactive dashboards with query-parameter filtering
- Exposed raw data views (HTML table + JSON endpoint)
- Implemented a basic A/B testing mechanism to compare user interactions
- Summarized insights and limitations for non-technical audiences

Key Features
- Interactive Analysis
- Sleep Quality vs Physical Activity visualization with filters
- Ranked comparison of occupations by mean sleep quality
- Dedicated /charts page for exploration

Raw Data Access

- HTML table view for manual inspection (/browse.html)
- JSON endpoint for programmatic access (/browse.json)
- Optional filtering via query parameters

Experimentation (A/B Testing)
- Simple A/B test that serves different versions of a call-to-action
- Tracks interaction counts to determine which variant performs better
- Demonstrates foundational experimentation and decision logic

Summary Metrics
- Average sleep quality and duration
- Proportion of records meeting a “poor sleep” threshold
- Notable correlations between lifestyle factors and sleep outcomes

Example Findings
- Higher physical activity levels are positively associated with sleep quality
- Sleep outcomes vary meaningfully across occupations
- A non-trivial portion of the dataset meets a defined “poor sleep” threshold

Tech Stack
   Python
   Flask
   pandas
   NumPy
   Matplotlib
   HTML/CSS (lightweight frontend)

Contact:
Sofia Roth
Data Science student interested in health analytics and applied data work
GitHub: https://github.com/siroth121
