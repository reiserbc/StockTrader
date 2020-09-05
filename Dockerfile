
FROM pytorch/pytorch

# Copy src files into container
COPY src .

# Install dependencies
RUN pip install -r requirements.txt

# Run server
CMD ["python", "live.py"]
