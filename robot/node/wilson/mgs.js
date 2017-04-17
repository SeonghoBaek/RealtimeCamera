var mongoose = require('mongoose');
var textSearch = require('mongoose-text-search');
var schema = require('./mgs_schema');

mongoose.connect('mongodb://wilson/log');

var con = mongoose.connection;
var db = con.db;

con.on('error', console.error.bind(console,'connection error:'));

con.once('open', function callback() {
    console.log('connection opened');
});

var Schema = mongoose.Schema;

var logSafe = {w:1};
var logSchema = new Schema(schema.LOG, {safe:logSafe});

logSchema.plugin(textSearch);
logSchema.index({"vector.text":"text", "vector.distance":1});

var logModel = mongoose.model('Log', logSchema);

var trSafe = {w:1};
var trlogSchema = new Schema(schema.TRLOG, {safe:trSafe});
var trlogModel = mongoose.model('TrLog', trlogSchema);

var clientSchema = new Schema(schema.CLIENT);
var clientModel = mongoose.model('Client', clientSchema);

var completeSchema = new Schema(schema.COMPLETE);
var completeModel = mongoose.model('Complete', completeSchema);

var errorSchema = new Schema(schema.ERROR);
var errorModel = mongoose.model('Error', errorSchema);

var transactLog = function(json) {
    clientModel.find({_id:json.cid}).count(function(err, count) {

        if (err) console.error(err);
        else if (count != 0) {

            // First write transaction log.
            var trLogDoc = new trlogModel({_id:json.cid});

            trLogDoc.save(function(err,res) {
                //db.collection('trLogs').insert({_id:json._id.id}, function(err,res) {
                if (err) {
                    // Transaction Ready.
                    console.log("TRANSACTION HOLD PID: " + process.pid + " Client: " + json.client);

                    setTimeout(function() {
                        transactLog(json);
                    }, 30);

                    return;
                } else {

                    logModel.find({"cid":json.cid, "segment":json.segment, "session":json.session}).count(function(err, logcnt) {
                        if (err) {
                            console.error(err);
                            // Last delete transaction log.
                            trlogModel.remove({_id:json.cid}, function(){});
                        }
                        else if (logcnt != 0) {
                            logModel.update({"cid":json.cid, "segment":json.segment, "session":json.session}, {$push:{vectors:json.vectors[0]}}, function(err, locDoc) {
                                if (err) console.error(err);
                                else {
                                    //console.log("Log for client " + logDoc.client + " Updated");
                                    /*
                                     var options = {

                                     project: '_id:1'                // do not include the `created` property
                                     , filter: {"vector.distance": { $gt: 10 }} // casts queries based on schema
                                     , limit: 5
                                     , lean: true
                                     };

                                     logModel.textSearch("mmcblk0: error", options, function(err, out) {
                                     var inspect = require('util').inspect;
                                     console.log(inspect(out, { depth: null }));
                                     });
                                     */
                                    console.log("UPDATE COMMIT PID: " + process.pid + " Client: " + json.client);
                                }

                                // Last delete transaction log.
                                trlogModel.remove({_id:json.cid}, function(){});
                            });
                        } else {

                            json.date = new Date();
                            json.result = 2; // unknown

                            var logDoc = new logModel(json);

                            console.log("START TRANSACTION PID: " + process.pid + " Client: " + json.client);

                            logDoc.save(function(err, doc) {
                                // Last delete transaction log.
                                trlogModel.remove({_id:json.cid}, function(){});

                                if (err) {
                                    console.error(err);
                                    console.log("REDO TRANSACTION PID: " + process.pid + " Client: " + json.client);
                                    transactLog(json);
                                }
                                else {
                                    console.log("SAVE COMMIT PID: " + process.pid + " Client: " + json.client);
                                }
                            });
                        }
                    });
                }
            });
        }
    });
};

var saveClient = function(json) {

    var clientDoc = new clientModel(json);

    // upsert
    clientDoc.save(function(err, doc, affected) {
        if (err) {
            console.error(err);
        }
        else {
            if (affected == 0)
                console.log("New Client Added");
            else
                console.log("Client Updated");
        }
    });

};

var saveComplete = function(json) {
    json.date = new Date();

    var completeRecord = new completeModel(json);

    completeRecord.save(function(err, doc) {
        if (err) console.error(err);
        else {
            console.log("Complete Document Created for " + doc.status.name);
        }
    });
};

var saveError = function(json) {
    json.date = new Date();

    var errorDoc = new errorModel(json);

    errorDoc.save(function(err, doc) {
        if (err) console.error(err);
        else {
            console.log("Error Document Created.");
        }
    });
};

exports.log = logModel;
exports.trlog = trlogModel;
exports.client = clientModel;
exports.complete = completeModel;
exports.error = errorModel;
exports.saveLog = transactLog;
exports.saveClient = saveClient;
exports.saveComplete = saveComplete;
exports.saveError = saveError;
