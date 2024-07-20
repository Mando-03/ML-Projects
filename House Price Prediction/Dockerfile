#this Dockerfile sets up a containerized environment for running a FastAPI application
# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

#copies all the files and directories from the current directory on the host machine to the /app directory in the Docker container.
COPY . /app

# Install any needed packages specified in requirements.txt,The --no-cache-dir prevents pip from caching the packages,which keep image smaller.
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8000 for the FastAPI app
EXPOSE 8000

# Define environment variable to disable buffering
ENV PYTHONUNBUFFERED=1

# Run the command to start the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
