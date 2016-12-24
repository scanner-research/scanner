var app = require('express')(),
    server = require('http').createServer(app),
    io = require('socket.io').listen(server),
    fs = require('fs');

server.listen(3000);

app.get('/', function (req, res) {
  res.sendfile(__dirname + '/index.html');
});

io.sockets.on('connection', function (socket) {
  socket.on('render-frame', function (data) {
    data.file = data.file.split(',')[1]; // Get rid of the data:image/png;base64 at the beginning of the file data
    var buffer = new Buffer(data.file, 'base64');
    var fileName = '/tmp/cam-' + data.cam + '-frame-' + data.frame + '.png';
    fs.writeFile(__dirname + fileName,
                 buffer.toString('binary'), 'binary',
                 (err) => {
                   if (err) {
                     console.log('Failed to save ' + fileName);
                     throw err;
                   }
                   console.log('Saved ' + fileName);
                 });
  });
  socket.on('calibration', function (data) {
    var fileName = '/tmp/calibration.json';
    fs.writeFile(__dirname + fileName,
                 JSON.stringify(data.calibration), 'ascii',
                 (err) => {
                   if (err) {
                     console.log('Failed to save ' + fileName);
                     throw err;
                   }
                   console.log('Saved ' + fileName);
                 });
  });
});
