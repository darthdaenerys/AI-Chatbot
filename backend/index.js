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


io.on('connection', (socket) => {
    console.log('User connected');
    socket.on('disconnect', () => {
        console.log('User disconnected');
    })
});


app.get('/', (req, res) => {
	res.send('Hello World!')
})

const pythonProcess = spawn('python', ['../chat.py']);

const onData=(chunk)=>{
    const generatedText=chunk.toString();
    // console.log('Sending chunk:', generatedText);
    io.emit('generatedText', generatedText);
}

const onError=(data)=>{
    console.error(`Error from Python script: ${data}`);
    res.status(500).json(data);
}

const onClose=(code)=>{
    io.emit('stop');
    res.status(200);
}

pythonProcess.stdout.on('data', onData);
pythonProcess.stderr.on('data', onError);
pythonProcess.on('close', onClose);

app.post('/message', (req, res) => {
    try{
        const chunk = req.body;
        pythonProcess.stdin.write(chunk.message+'\n');
        res.status(200).json("Success");

    }
    catch(error){
        console.log(error);
        res.status(500)
    }
});

server.listen(port, () => {
	console.log(`Example app listening on port ${port}`)
})