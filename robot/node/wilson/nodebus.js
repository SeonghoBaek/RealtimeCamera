/**
 * Created by major on 14. 8. 25.
 */
var net = require('net');
var fs = require('fs');
var cluster = require('cluster');
var numCore = require('os').cpus().length;

var PREFIX = "/var/tmp/";
var SERVER_ADDRESS = PREFIX + "defaultservice.service.global";

var XML_HEADER = '<?xml version="1.0" encoding="UTF-8"?>';
var GLOBAL_MESSAGE = 0;
var LOCAL_MESSAGE = 1;
var CLIENT_MESSAGE = 2;
var KILL_MESSAGE = 3;
var BC_LOCAL_MESSAGE = 4;
var BC_CLIENT_MESSAGE = 5;

var createHeader = function(xml, xmltype) {
    var header = new Buffer(8);
    var type = xmltype;

    //console.log(typeof xmltype);
    var length = xml.length;

    header[0]=type & (0xFF);
    type=type>>8;
    header[1]=type & (0xFF);
    type=type>>8;
    header[2]=type & (0xFF);
    type=type>>8;
    header[3]=type & (0xFF);

    header[4]=length & (0xFF);
    length=length>>8;
    header[5]=length & (0xFF);
    length=length>>8;
    header[6]=length & (0xFF);
    length=length>>8;
    header[7]=length & (0xFF);

    return header;
};

var join = function(nodeName, callback, multicore) {
    // Join to NodeBus Server
    if (multicore == true)
    {

        if (cluster.isMaster) {
            console.log('Join to Server');
            var client = net.connect({path: SERVER_ADDRESS});

            client.on('error', function (err) {
                console.log(err);
            });

            var joinXml = XML_HEADER + '<nodebus type="1" id="0" node="' + PREFIX + nodeName + '"/>';
            var header = createHeader(joinXml, LOCAL_MESSAGE);

            client.write(header);
            client.write(joinXml);
            client.end();

            try {
                fs.unlinkSync(PREFIX + nodeName);
            } catch (e) {}

            for (var i = 0; i < numCore - 1; i++) {
                cluster.fork();
            }

            cluster.on('exit', function (worker, code, signal) {
                console.log('worker' + worker.process.pid + ' dead');
            });
        } else {
            var nbusServer = net.createServer(function (sock) {

                var buff = new Buffer(4*4096);
                var start = 0;

                buff.fill(0);

                sock.on('data', function (data) {

                    data.copy(buff, start);

                    start += data.length;
                });

                sock.on('end', function() {
                    //console.log("Type: " + buff.readInt32LE(0) + " length: " + buff.readInt32LE(4));
                    callback(buff.slice(8, start));
                    start = 0;
                });
            });

            nbusServer.listen(PREFIX + nodeName);
        }

    } else {

        console.log('Join to Server');
        var client = net.connect({path: SERVER_ADDRESS});

        client.on('error', function (err) {
            console.log(err);
        });

        var joinXml = XML_HEADER + '<nodebus type="1" id="0" node="' + PREFIX + nodeName + '"/>';
        var header = createHeader(joinXml, LOCAL_MESSAGE);

        client.write(header);
        client.write(joinXml);
        client.end();

        try {
            fs.unlinkSync(PREFIX + nodeName);
        } catch (e) {}

        var nbusServer = net.createServer(function (sock) {
            sock.on('data', function (data) {
                callback(data.slice(8, data.length));
            });
        });

        nbusServer.listen(PREFIX + nodeName);
    }
};

var listen = function(nodeName, callback) {

        var nbusServer = net.createServer(function (sock) {
            sock.on('data', function (data) {
                callback(data.slice(8, data.length));
            });
        });

        nbusServer.listen(PREFIX + nodeName);
};

var cast = function(nodeName, data, type) {
    try {
        var client = net.connect({path:PREFIX + nodeName});
        var packet = data;
        var header = createHeader(packet, type);

        console.log(data.toString());
        client.write(header);
        client.write(packet);
        client.end();
    } catch (e) {
        console.log("write error: " + e + '\n');
    }
};

var send = function(nodeName, data) {
    try {
        console.log(data.toString());

        //var client = net.connect({path:PREFIX + nodeName});
        var client = net.createConnection(PREFIX + nodeName);

        client.on("connect", function(){
            var packet = data;
        
            client.write(packet);
            client.end();
        });
    } catch (e) {
        console.log("write error: " + e + '\n');
    }
};

exports.join = join;
exports.cast = cast;
exports.listen = listen;
exports.send = send;