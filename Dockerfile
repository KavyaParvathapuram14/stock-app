# Use the official Python base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /Users/kavyaparvathapuram/Desktop/swapna5307

# Copy project files into the container
COPY . /Users/kavyaparvathapuram/Desktop/swapna5307/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the default Streamlit port
EXPOSE 8080

# Run the Streamlit app
CMD ["streamlit", "run", "main.py", "--server.port=8080", "--server.address=0.0.0.0"]
