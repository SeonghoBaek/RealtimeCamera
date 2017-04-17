/**
 * Created by major on 14. 9. 29.
 */

var TRANSACTION =
{
    _id: Number,
    completed:Number
};

var LOG =
{
    cid:Number, // CLIENT reference key
    client:String,
    segment:Number,
    session:Number,
    tid:Number,
    file:String,
    version:Number,
    date:Date,
    vectors:
        [
            {
                distance:Number,
                version:Number,
                offset:Number,
                text:String,
                terms:
                    [
                        {
                            term:String,
                            freq:Number
                        }
                    ]
            }
        ]
};

var CLIENT =
{
    _id:Number, // Client Unique ID
    client:String,
    session:Number,
    file:String,
    tid:Number,
    status:String,  // For human readability
    similarity:Number,
    version:Number
};

var COMPLETE =
{
    client:String,
    tid:Number,
    session:Number,
    similarity:Number,
    result:Number,
    file:String,
    date:Date,
    version:Number
};

var ERROR =
{
    client:String,
    tid:Number,
    session:Number,
    date:Date,
    description:String,
    version:Number
};

var TESTCASE =
{
    _id:Number,    // Test case Unique ID
    xml:String,    // NodeBus Message XML
    description:String,
    version:Number
};

var TRLOG =
{
    _id: Number,
    completed:Number
};

exports.LOG = LOG;
exports.CLIENT = CLIENT;
exports.COMPLETE = COMPLETE;
exports.ERROR = ERROR;
exports.TESTCASE = TESTCASE;
exports.TRANSACTION = TRANSACTION;
exports.TRLOG = TRLOG;