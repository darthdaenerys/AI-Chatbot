const express = require('express')
const socketIO = require('socket.io');
var cors=require('cors');
const { spawn } = require('child_process');
const app = express()
const http=require('http');
const server=http.createServer(app)