#!/bin/bash

echo "Hello World"

# Change directory to client
cd client || exit

# Install npm dependencies
npm install

# Run the client
npm run dev