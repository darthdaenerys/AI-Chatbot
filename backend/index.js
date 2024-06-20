const express = require('express')
const socketIO = require('socket.io');
var cors=require('cors');
const { spawn } = require('child_process');
const app = express()
const http=require('http');
const server=http.createServer(app)
const io = socketIO(server,{
    cors: {
        origin: "*",
        methods: ["GET", "POST"]
    }
});
const port = 5000


app.use(cors());
app.use(express.json());
app.use(express.urlencoded({extended:true}));