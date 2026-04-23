const fs = require('fs');

try {
  const errBuf = fs.readFileSync('err.txt');
  let errTxt = errBuf.toString('utf16le');
  fs.writeFileSync('err_utf8.js_out.txt', errTxt, 'utf8');
  
  const outBuf = fs.readFileSync('out.json');
  let outTxt = outBuf.toString('utf16le');
  fs.writeFileSync('out_utf8.js_out.txt', outTxt, 'utf8');

  console.log("Done decoding");
} catch (e) {
  console.error("Error decoding:", e.message);
}
