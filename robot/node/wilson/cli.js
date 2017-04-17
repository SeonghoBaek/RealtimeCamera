/**
 * Created by major on 14. 8. 25.
 */
var net = require('net');
var fs = require('fs');
var nbus = require('nodebus');

var BIGNODE_ADDRESS = "bignode.node";
var NODE_NAME = "clinode.node";
var BASE_DIR = "../sample/";

var nodeCallback = function (data) {
    console.log(data +'\n');
}

// Join to NodeBus Server
nbus.join(NODE_NAME, nodeCallback, false);

process.stdin.resume();
process.stdin.setEncoding('utf8');
process.stdout.write('> ');

var input = '';

process.stdin.on('data', function(data) {

    input += data.toString().trim().toLowerCase();

    if (input == "exit") {
        process.stdin.emit('end');
    } else {
        //if (data.toString().trim().slice(-1) == ";") {
        if (input.trim().length > 0) {
            //var cmdline = input.substr(0, input.indexOf(';'));
            var cmdline = input;
            var cmds = cmdline.split(" ");

            var cmd = cmds[0];
            var arg = cmds[1];

            if (cmd == 'run') {
                if (arg == null) {
                    process.stdout.write('run [file name]\n');
                } else {
                    fs.readFile(BASE_DIR + arg, function (err, data) {
                        if (err) process.stdout.write(err.toString() + '\n');
                        if (data) {
                            process.stdout.write(data.toString() + '\n');
                        }

                        nbus.cast(BIGNODE_ADDRESS, cmdline, BC_CLIENT_MESSAGE);

                        process.stdout.write('> ');
                    });
                }
            } else {
                process.stdout.write('> ');
            }

            input = '';
        } else if (data.trim().length > 0) {
            process.stdout.write('-> ');
        } else {
            process.stdout.write('> ');
        }
    }
});

process.stdin.on('end', function() {
    process.stdout.write('end\n');
    process.exit(0);
});
