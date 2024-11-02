# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory to /app
WORKDIR /application

# Install Jupyter Notebook
RUN pip install jupyter

# Test if this is necessary
COPY requirements.txt .

# Install necessary packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory content into the app folder
COPY . .

# Make port 8888 available to the world outside this container
EXPOSE 8888

# Run Jupyter Notebook when the container launches
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]