# Download python from dockerhub
FROM python:3.8.10

#Working directory
WORKDIR /home/dloader/Documents/GitHub/tutorial-hell/FourthBrainAI/nlp-deployment-tutorial
# Copy requirements to the container
COPY requirements.txt .

# Install dependencies
RUN pip  install -r requirements.txt 

# Copy the flask app to the working dir
COPY src/ .

# Expose ports
EXPOSE 8080

# Call application
CMD ["python3", "app.py", "serve"]