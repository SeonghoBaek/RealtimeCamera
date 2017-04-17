/**
 * Created by major on 14. 11. 19.
 */

var GET_CLIENT_LIST = 'nbus.list';
var RUN_TEST_CASE = 'nbus.run';
var STOP_TEST_CASE = 'nbus.stop';

// socket.io.js should be imported first.

function send(evt, data) {
    if (this.socket) {
        this.socket.emit(evt, data);
    }
}

function NBUS(url, callback) {
    this.socket = io.connect(url, {'reconnect':true, 'resource': 'socket.io'});

    this.send = send;

    this.socket.on(GET_CLIENT_LIST, function(data) {
        callback(GET_CLIENT_LIST, data);
    });

    this.socket.on(RUN_TEST_CASE, function(data) {
        callback(RUN_TEST_CASE, data);
    });

    this.socket.on(STOP_TEST_CASE, function(data) {
        callback(STOP_TEST_CASE, data);
    });
}