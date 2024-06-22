# Use the official Node.js base image
FROM node:18-alpine

# Set the working directory inside the container
WORKDIR /app

# Install git to clone the repository
RUN apk add --no-cache git

# Clone the Pokémon Showdown repository
RUN git clone https://github.com/smogon/pokemon-showdown.git .

# Install the app dependencies
RUN npm install

# Expose the port that the app will run on
EXPOSE 8000

# Start the app
CMD ["node", "pokemon-showdown", "start", "--no-security"]
