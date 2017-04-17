/**
 * Created by major on 14. 11. 19.
 */
var mongoose = require('mongoose');
var textSearch = require('mongoose-text-search');
var nbus = require('./nodebus');
var schema = require('./mgs_schema');

var NODE_NAME = "socketio.node";
var BIGNODE_NAME = "bignode.node";

var http = require('http');
var socketio =  require('socket.io');

// Create Socket I/O Server
var serverPort = 8089;

var GET_CLIENT_LIST = 'nbus.list';
var RUN_TEST_CASE = 'nbus.run';
var STOP_TEST_CASE = 'nbus.stop';

var DB_NAME = 'mongodb://wilson/log';

mongoose.connect(DB_NAME);

var con = mongoose.connection;
var db = con.db;

con.on('error', console.error.bind(console,'connection error:'));
con.once('open', function callback() {
    console.log('connection opened');
});

var Schema = mongoose.Schema;

var logSchema = new Schema(mgs_schema.LOG);

logSchema.plugin(textSearch);
logSchema.index({"vector.text":"text", "vector.distance":1});

var logModel = mongoose.model('Log', logSchema);

var clientSchema = new Schema(mgs_schema.CLIENT);
var clientModel = mongoose.model('Client', clientSchema);

var completeSchema = new Schema(mgs_schema.COMPLETE);
var completeModel = mongoose.model('Complete', completeSchema);

var errorSchema = new Schema(mgs_schema.ERROR);
var errorModel = mongoose.model('Error', errorSchema);

var server = http.createServer(function(request, response) {}).listen(serverPort, function() {
    console.log('Socket I/O Server Running at Port ' + serverPort);
});

var sio = socketio.listen(server);

sio.sockets.on('connection', function(socket) {
    console.log('socket I/O client connected');

    socket.on(GET_CLIENT_LIST, function(data) {
        console.log('event GET_CLIENT_LIST arrived');
        var lists = getClientList();

        socket.emit(GET_CLIENT_LIST, lists);
    });

    socket.on(RUN_TEST_CASE, function(data) {

    });

    socket.on(STOP_TEST_CASE, function(data) {

    });
});

var getClientList = function() {
    var lists = [{'name':'client1'},{'name':'client2'}];

    clientModel.find({}, function(err, data) {
        if (err) console.log(err);
        else {
            console.log(data);
        }
    });

    return lists;
};

var nodeCallback = function (data) {
    console.log(data +'\n');
};

nbus.join(NODE_NAME, nodeCallback, false);
